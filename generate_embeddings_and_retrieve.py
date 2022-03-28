"""
Combine the functionalities of the two scripts
`generate_dense_embeddings.py` + `dense_retriever.py`. Note that some behavior
has been disabled, e.g., `cfg.kilt_out_file` in the dense retriever script.
"""


import logging
import math
import sys
from typing import List, Tuple, Generator

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from torch import nn

from dpr.data.biencoder_data import BiEncoderPassage
from dpr.models import init_biencoder_components
from dpr.options import set_cfg_params_from_state, setup_cfg_gpu, setup_logger
from dpr.utils.dist_utils import gather, synchronize

from dpr.utils.data_utils import Tensorizer
from dpr.utils.model_utils import (
    setup_for_distributed_mode,
    get_model_obj,
    load_states_from_checkpoint,
)

# Locally import from those two scripts
from generate_dense_embeddings import gen_ctx_vectors as legacy_gen_ctx_vectors
from dense_retriever import LocalFaissRetriever, validate, save_results


logger = logging.getLogger()
setup_logger(logger)


def gen_ctx_vectors(
    cfg: DictConfig,
    ctx_rows: List[Tuple[object, BiEncoderPassage]],
    model: nn.Module,
    tensorizer: Tensorizer,
    buffer_size: int,
    insert_title: bool = True,
):
    """
    Convert the legacy `gen_ctx_vectors` into a generator.
    """
    num_batches = math.ceil(len(ctx_rows) / buffer_size)
    for i in range(num_batches):
        start = i * buffer_size
        end = start + buffer_size
        results = legacy_gen_ctx_vectors(
            cfg,
            ctx_rows[start:end],
            model,
            tensorizer,
            insert_title,
        )
        for res in results:
            yield res


class DistributedFaissRetriever(LocalFaissRetriever):
    """
    Does passage retrieving over the provided index and question encoder.
    This implementation does searching in document index locally,
    then aggregating the results from multiple processes in the distributed
    settings.
    """
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = cfg

    def index_encoded_data(
        self,
        embeddings: Generator[Tuple[str, np.ndarray], None, None],
        buffer_size: int,
        id_prefix,
    ):
        buffer = []
        for doc_id, embs in embeddings:
            if not str(doc_id).startswith(id_prefix):
                doc_id = id_prefix + str(doc_id)
            buffer.append((doc_id, embs))
            if 0 < buffer_size == len(buffer):
                self.index.index_data(buffer)
                buffer = []
        self.index.index_data(buffer)
        logger.info("Data indexing completed.")

    def get_top_docs(
        self, query_vectors: np.array, top_docs: int = 100
    ) -> List[Tuple[List[object], List[float]]]:
        """
        Do searching in document indices lcoally then aggregate from multiple
        processes.
        """
        # Search locally
        local_results = super().get_top_docs(query_vectors, top_docs)
        assert len(local_results) == len(query_vectors)

        # Gather from multiple processes
        results = gather(self.cfg, [local_results])[0]

        # Aggregate
        tot_results = [
            ([], []) for _ in range(len(query_vectors))
        ] # list of tuples (passage_ids, scores), one for each question input
        larger_is_better = None

        for results_i in results:
            for i, (passage_ids, scores) in enumerate(results_i):
                # Determine whether the scores are better when larger or smaller
                # Assume top_docs > 1
                larger_is_better_i = (scores.argmax() == 0)
                if larger_is_better is not None:
                    assert larger_is_better == larger_is_better_i
                else:
                    larger_is_better = larger_is_better_i

                tot_results[i][0].extend(passage_ids)
                tot_results[i][1].extend(scores)

        # Post-process
        for i, (passage_ids, scores) in enumerate(tot_results):
            idxs = np.argsort(scores)
            if larger_is_better:
                idxs = idxs[::-1]
            idxs = idxs[:top_docs]

            passage_ids = [passage_ids[i] for i in idxs]
            scores = [scores[i] for i in idxs]
            tot_results[i] = (passage_ids, scores)
        return tot_results


@hydra.main(config_path="conf", config_name="gen_embs_and_retrieve")
def main(cfg: DictConfig):
    # Sanity check
    assert cfg.model_file, "Please specify encoder checkpoint as model_file param"
    assert cfg.ctx_src, "Please specify passages source as ctx_src param"
    assert cfg.qa_dataset, "Please specify qa_dataset to use"

    # Set up cfg
    cfg = setup_cfg_gpu(cfg)
    saved_state = load_states_from_checkpoint(cfg.model_file)
    set_cfg_params_from_state(saved_state.encoder_params, cfg)
    logger.info("CFG:")
    logger.info("%s", OmegaConf.to_yaml(cfg))

    # Initialize tensorizer and model
    tensorizer, encoder, _ = init_biencoder_components(
        cfg.encoder.encoder_model_type, cfg, inference_only=True)
    encoder, _ = setup_for_distributed_mode(
        encoder,
        None,
        cfg.device,
        cfg.n_gpu,
        cfg.local_rank,
        cfg.fp16,
        cfg.fp16_opt_level,
    )
    encoder.eval()

    # Load weights from the model file
    encoder = get_model_obj(encoder)
    vector_size = encoder.question_model.get_out_size()
    logger.info("Loading saved model state ...")
    encoder.load_state_dict(saved_state.model_dict, strict=False)

    # ******************** GENERATE CONTEXT EMBEDDINGS ********************

    # Read data source
    logger.info("reading data source: %s", cfg.ctx_src)
    ctx_src = hydra.utils.instantiate(cfg.ctx_sources[cfg.ctx_src])
    all_passages_dict = {}
    ctx_src.load_data_to(all_passages_dict)
    all_passages = [(k, v) for k, v in all_passages_dict.items()]

    # Get statistics
    if cfg.local_rank == -1:
        start_idx, end_idx = 0, len(all_passages)
    else:
        shard_size = math.ceil(len(all_passages) / cfg.distributed_world_size)
        start_idx = cfg.local_rank * shard_size
        end_idx = start_idx + shard_size

    # Start generating embeddings
    logger.info(
        "Producing encodings for passages range: %d to %d (out of total %d)",
        start_idx,
        end_idx,
        len(all_passages),
    )
    shard_passages = all_passages[start_idx:end_idx]
    ctx_embs = gen_ctx_vectors(
        cfg, shard_passages, encoder.ctx_model, tensorizer, True)

    # *************** GENERATE QUESTION EMBEDDINGS ***************

    ds_key = cfg.qa_dataset
    logger.info("qa_dataset: %s", ds_key)
    qa_src = hydra.utils.instantiate(cfg.datasets[ds_key])
    qa_src.load_data()

    # get questions & answers
    questions = []
    question_answers = []
    for i in range(len(qa_src)):
        qa_sample = qa_src[i]
        question, answers = qa_sample.query, qa_sample.answers
        questions.append(question)
        question_answers.append(answers)
    logger.info("questions len %d", len(questions))

    # Initialize indexer
    index = hydra.utils.instantiate(cfg.indexers[cfg.indexer])
    logger.info("Local Index class %s ", type(index))
    index_buffer_sz = index.buffer_size
    index.init_index(vector_size)
    retriever = DistributedFaissRetriever(cfg, encoder.question_model, cfg.batch_size, tensorizer, index)

    logger.info("Using special token %s", qa_src.special_query_token)
    questions_tensor = retriever.generate_question_vectors(questions, query_token=qa_src.special_query_token)

    if qa_src.selector:
        logger.info("Using custom representation token selector")
        retriever.selector = qa_src.selector

    # index all encoded passages
    retriever.index_encoded_data(ctx_embs, index_buffer_sz, id_prefix=ctx_src.id_prefix)

    # get top k results
    top_results_and_scores = retriever.get_top_docs(questions_tensor.numpy(), cfg.n_docs)
    questions_doc_hits = validate(
        all_passages_dict,
        question_answers,
        top_results_and_scores,
        cfg.validation_workers,
        cfg.match,
    )

    if cfg.local_rank in [-1, 0]:
        save_results(
            all_passages_dict,
            questions,
            question_answers,
            top_results_and_scores,
            questions_doc_hits,
            cfg.out_file,
        )
    synchronize()


if __name__ == "__main__":
    logger.info("Sys.argv: %s", sys.argv)
    hydra_formatted_args = []
    # convert the cli params added by torch.distributed.launch into Hydra format
    for arg in sys.argv:
        if arg.startswith("--"):
            hydra_formatted_args.append(arg[len("--") :])
        else:
            hydra_formatted_args.append(arg)
    logger.info("Hydra formatted Sys.argv: %s", hydra_formatted_args)
    sys.argv = hydra_formatted_args

    main()
