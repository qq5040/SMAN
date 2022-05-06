import argparse
import math
import os
import torch
from torch.nn import DataParallel
from torch.optim import Optimizer
import transformers
from transformers import AdamW
from transformers import BertTokenizer

from models.sman import SMAN
from models.entities import Dataset
from models.evaluator import Evaluator
from models.input_reader import JsonInputReader, BaseInputReader
from models.loss import ModelLoss, Loss
from tqdm import tqdm
from models.sampling import Sampler
from models.trainer import BaseTrainer

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))


class SMANTrainer(BaseTrainer):
    """ Joint entity and relation extraction training and evaluation """

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)

        # seed = args.seed
        # random.seed(seed)
        # np.random.seed(seed)
        # os.environ['PYTHONHASHSEED'] = str(seed)
        # torch.manual_seed(seed)
        # torch.cuda.manual_seed(seed)

        # byte-pair encoding
        self._tokenizer = BertTokenizer.from_pretrained(args.tokenizer_path,
                                                        do_lower_case=args.lowercase,
                                                        cache_dir=args.cache_path)

        # path to export relation extraction examples to
        self._examples_path = os.path.join(self._log_path, 'examples_%s_%s_epoch_%s.html')

        # sampler (create and batch training/evaluation samples)
        self._sampler = Sampler(processes=args.sampling_processes, limit=args.sampling_limit)

        self.mi_F1 = [0., 0.]
        self.ma_F1 = [0., 0.]
        self.best_F1 = [0., 0.]

    def train(self, train_path: str, valid_path: str, types_path: str, input_reader_cls: BaseInputReader):
        args = self.args
        train_label, valid_label = 'train', 'valid'

        self._logger.info("Datasets: %s, %s" % (train_path, valid_path))

        # create log csv files
        self._init_train_logging(train_label)
        self._init_eval_logging(valid_label)

        # read datasets
        input_reader = input_reader_cls(types_path, self._tokenizer, self._logger)
        input_reader.read({train_label: train_path, valid_label: valid_path})
        self._log_datasets(input_reader)

        train_dataset = input_reader.get_dataset(train_label)
        train_sample_count = train_dataset.document_count

        updates_epoch = train_sample_count // args.train_batch_size

        updates_total = updates_epoch * args.epochs

        validation_dataset = input_reader.get_dataset(valid_label)

        self._logger.info("Updates per epoch: %s" % updates_epoch)
        self._logger.info("Updates total: %s" % updates_total)

        max_entity_encode_size = max(train_dataset.max_entity_encode_size(), validation_dataset.max_entity_encode_size())

        max_entity_token_size = max(train_dataset.max_entity_token_size(), validation_dataset.max_entity_token_size())

        max_relation_distance = max(train_dataset.max_relation_distance(), validation_dataset.max_relation_distance())

        self._logger.info("max_entity_token_size: %s" % max_entity_token_size)
        self._logger.info("max_entity_encode_size: %s" % max_entity_encode_size)
        self._logger.info("max_relation_distance: %s" % max_relation_distance)

        model = SMAN.from_pretrained(self.args.model_path,
                                     cache_dir=self.args.cache_path,
                                     cls_token=self._tokenizer.convert_tokens_to_ids('[CLS]'),
                                     sep_token=self._tokenizer.convert_tokens_to_ids('[SEP]'),
                                     relation_types=input_reader.relation_type_count - 1,
                                     entity_types=input_reader.entity_type_count,
                                     max_pairs=self.args.max_pairs,
                                     prop_drop=self.args.dropout,
                                     size_embedding=self.args.size_embedding,
                                     freeze_transformer=self.args.freeze_transformer,
                                     encoder_layers=self.args.encoder_layers,
                                     encoder_heads=self.args.encoder_heads,
                                     dropout=self.args.dropout,
                                     device=self._device)

        model.to(self._device)

        # create optimizer
        optimizer_params = self._get_optimizer_params(model)
        optimizer = AdamW(optimizer_params, lr=args.lr, weight_decay=args.weight_decay, correct_bias=False)

        # create scheduler
        scheduler = transformers.get_linear_schedule_with_warmup(optimizer,
                                                                 num_warmup_steps=args.lr_warmup * updates_total,
                                                                 num_training_steps=updates_total)
        # create loss function
        rel_criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
        entity_criterion = torch.nn.CrossEntropyLoss(reduction='none')
        compute_loss = ModelLoss(rel_criterion, entity_criterion, model, optimizer, scheduler, args.max_grad_norm)

        # eval validation set
        if args.init_eval:
            self._eval(model, validation_dataset, input_reader, 0, updates_epoch)

        # train
        for epoch in range(args.epochs):
            # train epoch
            self._train_epoch(model, compute_loss, optimizer, train_dataset, updates_epoch, epoch,
                              input_reader.context_size, input_reader.relation_type_count)

            # eval validation sets
            if args.final_eval or (epoch == args.epochs - 1):
                self._eval(model, validation_dataset, input_reader, epoch + 1, updates_epoch)

            # save epoch model
            if self.mi_F1[0] > 90 and self.mi_F1[1] > 70:
            # if self.ma_F1[0] > 90 and self.ma_F1[1] > 80:
            # if self.mi_F1[0] > 71 and self.mi_F1[1] > 53:
                if (self.mi_F1[0] + self.mi_F1[1]) - (self.best_F1[0] + self.best_F1[1]) >= 0:
                    self.best_F1[0] = self.mi_F1[0]
                    self.best_F1[1] = self.mi_F1[1]

                    extra = dict(epoch=args.epochs, updates_epoch=updates_epoch, epoch_iteration=0)
                    global_iteration = args.epochs * updates_epoch
                    self._save_model(self._save_path, model, global_iteration,
                                     optimizer=optimizer if self.args.save_optimizer else None, extra=extra,
                                     include_iteration=False, name='Train_epoch_'+str(epoch))
                    print('model save finshed!')

        self._logger.info("NER-best-F1: %s" % self.best_F1[0])
        self._logger.info("REL-best-F1: %s" % self.best_F1[1])

        self._logger.info("Logged in: %s" % self._log_path)
        self._logger.info("Saved in: %s" % self._save_path)

        self._sampler.join()

    def eval(self, dataset_path: str, types_path: str, input_reader_cls: BaseInputReader):
        args = self.args
        dataset_label = 'test'

        self._logger.info("Dataset: %s" % dataset_path)

        # create log csv files
        self._init_eval_logging(dataset_label)

        # read datasets
        input_reader = input_reader_cls(types_path, self._tokenizer, self._logger)
        input_reader.read({dataset_label: dataset_path})
        self._log_datasets(input_reader)

        model = SMAN.from_pretrained(self.args.model_path,
                                     cache_dir=self.args.cache_path,
                                     cls_token=self._tokenizer.convert_tokens_to_ids('[CLS]'),
                                     sep_token=self._tokenizer.convert_tokens_to_ids('[SEP]'),
                                     relation_types=input_reader.relation_type_count - 1,
                                     entity_types=input_reader.entity_type_count,
                                     max_pairs=self.args.max_pairs,
                                     prop_drop=self.args.dropout,
                                     size_embedding=self.args.size_embedding,
                                     freeze_transformer=self.args.freeze_transformer,
                                     encoder_layers=self.args.encoder_layers,
                                     encoder_heads=self.args.encoder_heads,
                                     dropout=self.args.dropout,
                                     device=self._device)

        model.to(self._device)

        # evaluate
        self._eval(model, input_reader.get_dataset(dataset_label), input_reader)
        self._logger.info("Logged in: %s" % self._log_path)

        self._sampler.join()

    def _train_epoch(self, model: torch.nn.Module, compute_loss: Loss, optimizer: Optimizer, dataset: Dataset,
                     updates_epoch: int, epoch: int, context_size: int, rel_type_count: int):
        self._logger.info("Train epoch: %s" % epoch)

        # randomly shuffle data
        order = torch.randperm(dataset.document_count)
        sampler = self._sampler.create_train_sampler(dataset, self.args.train_batch_size, self.args.max_span_size,
                                                     context_size, self.args.neg_entity_count,
                                                     self.args.neg_relation_count, order=order, truncate=True)

        model.zero_grad()

        iteration = 0
        total = dataset.document_count // self.args.train_batch_size

        for batch in tqdm(sampler, total=total, desc='Train epoch %s' % epoch):

            torch.cuda.empty_cache()

            model.train()
            batch = batch.to(self._device)

            # relation types to one-hot encoding
            rel_types_onehot = torch.zeros([batch.rel_types.shape[0], batch.rel_types.shape[1],
                                            rel_type_count], dtype=torch.float32).to(self._device)
            rel_types_onehot.scatter_(2, batch.rel_types.unsqueeze(2), 1)
            rel_types_onehot = rel_types_onehot[:, :, 1:]  # all zeros for 'none' relation

            # forward step
            entity_logits, rel_logits = model(batch.encodings, batch.ctx_masks, batch.entity_masks, batch.eup_masks, batch.edown_masks, batch.entity_types,
                                              batch.entity_sizes, batch.entity_start, batch.entity_end, batch.rels, batch.rel_masks, batch.rup_masks, batch.rdown_masks)

            # compute loss and optimize parameters
            batch_loss = compute_loss.compute(rel_logits, rel_types_onehot, entity_logits,
                                              batch.entity_types, batch.rel_sample_masks, batch.entity_sample_masks)

            # logging
            iteration += 1
            global_iteration = epoch * updates_epoch + iteration

            if global_iteration % self.args.train_log_iter == 0:
                self._log_train(optimizer, batch_loss, epoch, iteration, global_iteration, dataset.label)

        return iteration

    def _eval(self, model: torch.nn.Module, dataset: Dataset, input_reader: JsonInputReader,
              epoch: int = 0, updates_epoch: int = 0, iteration: int = 0):
        self._logger.info("Evaluate: %s" % dataset.label)

        if isinstance(model, DataParallel):
            # currently no multi GPU support during evaluation
            model = model.module

        # create evaluator
        evaluator = Evaluator(dataset, input_reader, self._tokenizer,
                              self.args.rel_filter_threshold, self.args.example_count,
                              self._examples_path, epoch, dataset.label)

        # create batch sampler
        sampler = self._sampler.create_eval_sampler(dataset, self.args.eval_batch_size, self.args.max_span_size,
                                                    input_reader.context_size, truncate=False)

        with torch.no_grad():
            model.eval()

            # iterate batches
            total = math.ceil(dataset.document_count / self.args.eval_batch_size)
            for batch in tqdm(sampler, total=total, desc='Evaluate epoch %s' % epoch):

                torch.cuda.empty_cache()

                # move batch to selected device
                batch = batch.to(self._device)

                # run model (forward pass)
                entity_clf, rel_clf, rels = model(batch.encodings, batch.ctx_masks, batch.entity_masks, batch.eup_masks, batch.edown_masks,
                                                  batch.entity_sizes,  batch.entity_start, batch.entity_end, batch.entity_spans, batch.entity_sample_masks,
                                                  evaluate=True)

                # evaluate batch
                evaluator.eval_batch(entity_clf, rel_clf, rels, batch)

        global_iteration = epoch * updates_epoch + iteration
        ner_eval, rel_eval, rel_ner_eval = evaluator.compute_scores()
        self._log_eval(*ner_eval, *rel_eval, *rel_ner_eval,
                       epoch, iteration, global_iteration, dataset.label)
        self._logger.info("Evaluate_F1result: NER-micro:%.2f ;NER-macro:%.2f"
                          % (float(ner_eval[2]), float(ner_eval[5])))
        self._logger.info("                   REL-micro:%.2f ;REL-macro:%.2f"
                          % (float(rel_eval[2]), float(rel_eval[5])))
        self._logger.info("                   RELner-micro:%.2f ;RELner-macro:%.2f"
                          % (float(rel_ner_eval[2]), float(rel_ner_eval[5])))

        self.mi_F1 = (float(ner_eval[2]), float(rel_eval[2]))
        self.ma_F1 = (float(ner_eval[5]), float(rel_eval[5]))

        if self.args.store_examples:
            evaluator.store_examples()

    def _get_optimizer_params(self, model):
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_params = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

        return optimizer_params

    def _log_train(self, optimizer: Optimizer, loss: float, epoch: int,
                   iteration: int, global_iteration: int, label: str):
        # average loss
        avg_loss = loss / self.args.train_batch_size
        # get current learning rate
        lr = self._get_lr(optimizer)[0]

        # log to tensorboard
        self._log_tensorboard(label, 'loss', loss, global_iteration)
        self._log_tensorboard(label, 'loss_avg', avg_loss, global_iteration)
        self._log_tensorboard(label, 'lr', lr, global_iteration)

        # log to csv
        self._log_csv(label, 'loss', loss, epoch, iteration, global_iteration)
        self._log_csv(label, 'loss_avg', avg_loss, epoch, iteration, global_iteration)
        self._log_csv(label, 'lr', lr, epoch, iteration, global_iteration)

    def _log_eval(self, ner_prec_micro: float, ner_rec_micro: float, ner_f1_micro: float,
                  ner_prec_macro: float, ner_rec_macro: float, ner_f1_macro: float,

                  rel_prec_micro: float, rel_rec_micro: float, rel_f1_micro: float,
                  rel_prec_macro: float, rel_rec_macro: float, rel_f1_macro: float,

                  rel_ner_prec_micro: float, rel_ner_rec_micro: float, rel_ner_f1_micro: float,
                  rel_ner_prec_macro: float, rel_ner_rec_macro: float, rel_ner_f1_macro: float,
                  epoch: int, iteration: int, global_iteration: int, label: str):

        # log to tensorboard
        self._log_tensorboard(label, 'eval/ner_prec_micro', ner_prec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_recall_micro', ner_rec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_f1_micro', ner_f1_micro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_prec_macro', ner_prec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_recall_macro', ner_rec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_f1_macro', ner_f1_macro, global_iteration)

        self._log_tensorboard(label, 'eval/rel_prec_micro', rel_prec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_recall_micro', rel_rec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_f1_micro', rel_f1_micro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_prec_macro', rel_prec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_recall_macro', rel_rec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_f1_macro', rel_f1_macro, global_iteration)

        self._log_tensorboard(label, 'eval/rel_ner_prec_micro', rel_ner_prec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_ner_recall_micro', rel_ner_rec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_ner_f1_micro', rel_ner_f1_micro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_ner_prec_macro', rel_ner_prec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_ner_recall_macro', rel_ner_rec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_ner_f1_macro', rel_ner_f1_macro, global_iteration)

        # log to csv
        self._log_csv(label, 'eval', ner_prec_micro, ner_rec_micro, ner_f1_micro,
                      ner_prec_macro, ner_rec_macro, ner_f1_macro,

                      rel_prec_micro, rel_rec_micro, rel_f1_micro,
                      rel_prec_macro, rel_rec_macro, rel_f1_macro,

                      rel_ner_prec_micro, rel_ner_rec_micro, rel_ner_f1_micro,
                      rel_ner_prec_macro, rel_ner_rec_macro, rel_ner_f1_macro,
                      epoch, iteration, global_iteration)

    def _log_datasets(self, input_reader):
        self._logger.info("Relation type count: %s" % input_reader.relation_type_count)
        self._logger.info("Entity type count: %s" % input_reader.entity_type_count)

        self._logger.info("Entities:")
        for e in input_reader.entity_types.values():
            self._logger.info(e.verbose_name + '=' + str(e.index))

        self._logger.info("Relations:")
        for r in input_reader.relation_types.values():
            self._logger.info(r.verbose_name + '=' + str(r.index))

        for k, d in input_reader.datasets.items():
            self._logger.info('Dataset: %s' % k)
            self._logger.info("Document count: %s" % d.document_count)
            self._logger.info("Relation count: %s" % d.relation_count)
            self._logger.info("Entity count: %s" % d.entity_count)

        self._logger.info("Context size: %s" % input_reader.context_size)

    def _init_train_logging(self, label):
        self._add_dataset_logging(label,
                                  data={'lr': ['lr', 'epoch', 'iteration', 'global_iteration'],
                                        'loss': ['loss', 'epoch', 'iteration', 'global_iteration'],
                                        'loss_avg': ['loss_avg', 'epoch', 'iteration', 'global_iteration']})

    def _init_eval_logging(self, label):
        self._add_dataset_logging(label,
                                  data={'eval': ['ner_prec_micro', 'ner_rec_micro', 'ner_f1_micro',
                                                 'ner_prec_macro', 'ner_rec_macro', 'ner_f1_macro',
                                                 'rel_prec_micro', 'rel_rec_micro', 'rel_f1_micro',
                                                 'rel_prec_macro', 'rel_rec_macro', 'rel_f1_macro',
                                                 'rel_ner_prec_micro', 'rel_ner_rec_micro', 'rel_ner_f1_micro',
                                                 'rel_ner_prec_macro', 'rel_ner_rec_macro', 'rel_ner_f1_macro',
                                                 'epoch', 'iteration', 'global_iteration']})
