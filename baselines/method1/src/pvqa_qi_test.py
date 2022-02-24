from logging import NullHandler
import re
from tarfile import NUL
from torch.utils.tensorboard import SummaryWriter
import os
import collections

import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from param import args

from pretrain.qa_answer_table import load_lxmert_qa
from tasks.pvqa_model_autoencoder import PVQAAutoencoderModel
from tasks.pvqa_data import PVQADataset, PVQATorchDataset, PVQAEvaluator
baseUrl = 'drive/MyDrive/PathVQA'
checkpoint_dir = baseUrl+"/checkpoint_LXRT.pth"
load_dir = baseUrl+"/checkpoint"
temp_checkpoint_save_dir = baseUrl+"/checkpoint_with_LXRT.pth"
adv_model_dir = baseUrl + \
    "/checkpoint_phase3_with_initial_lxrt_model_without_phase2_weights.pth"
LogFile = baseUrl+'/debug.txt'
# f = open(os.path.dirname(os.path.abspath(__file__)+"/"+writeFileName, "w+")
f = open(LogFile, "w+")
# f.close()
LogFile = open(LogFile, "r+")

startFrom = 'B'  # M - middle ,   B - beginning

# default `log_dir` is "runs" - we'll be more specific here
# writer = SummaryWriter(baseUrl+'runs/Pathvqa_experiment_qa')

DataTuple = collections.namedtuple("DataTuple", 'dataset loader evaluator')
valid_bs = 256


def get_data_tuple(splits: str, bs: int, shuffle=False, drop_last=False) -> DataTuple:
    dset = PVQADataset(splits)
    tset = PVQATorchDataset(dset)
    evaluator = PVQAEvaluator(dset)
    data_loader = DataLoader(
        tset, batch_size=bs, shuffle=shuffle, num_workers=args.num_workers,
        drop_last=drop_last, pin_memory=True
    )
    return DataTuple(dataset=dset, loader=data_loader, evaluator=evaluator)


class PVQA:
    def __init__(self):
        # datasets

        self.train_tuple = get_data_tuple(
            splits=args.train, bs=args.batch_size, shuffle=True, drop_last=True
        )
        if args.valid != "":
            self.valid_tuple = get_data_tuple(
                splits=args.valid, bs=valid_bs,
                shuffle=False, drop_last=False
            )
        else:
            self.valid_tuple = None

        # Model
        self.model = self.newLoadModel()['saved_full_model']

        # # Load pre-trained weights
        # if args.load_lxmert is not None:
        #     print(args.load_lxmert)
        #     if(startFrom == 'B'):
        #         self.model.lxrt_encoder.load(args.load_lxmert)
        #     # else:
        #     #     self.model.lxrt_encoder.load(load_dir)
        # if args.load_lxmert_qa is not None:
        #     print(args.load_lxmert_qa)

        #     load_lxmert_qa(args.load_lxmert_qa, self.model,
        #                    label2ans=self.train_tuple.dataset.label2ans)
        # GPU options
        self.model = self.model.cuda()
        # if args.multiGPU:
        #     self.model.lxrt_encoder.multi_gpu()

        # Loss and Optimizer
        # self.bce_loss = nn.BCEWithLogitsLoss()
        # if 'bert' in args.optim:
        #     batch_per_epoch = len(self.train_tuple.loader)
        #     t_total = int(batch_per_epoch * args.epochs)
        #     print("BertAdam Total Iters: %d" % t_total)
        #     from lxrt.optimization import BertAdam
        #     self.optim = BertAdam(list(self.model.parameters()),
        #                           lr=args.lr,
        #                           warmup=0.1,
        #                           t_total=t_total)
        # else:
        #     self.optim = args.optimizer(self.model.parameters(), args.lr)

        # Output Directory
        self.output = args.output
        os.makedirs(self.output, exist_ok=True)

    def train(self, train_tuple, eval_tuple):
        dset, loader, evaluator = train_tuple
        iter_wrapper = (lambda x: tqdm(x, total=len(loader))
                        ) if args.tqdm else (lambda x: x)

        best_valid = 0.

        lastEpoch = -1
        running_loss = 0
        print('loss and epoch reset')
        if(startFrom == "M"):
            print('loading from saved model..')
            checkpoint = self.getLastEpoch()
            lastEpoch = checkpoint['epoch']
            running_loss = checkpoint['loss']
            print("last epoch :" + str(lastEpoch))
            print('loss')
            print(running_loss)

        start_epoch = lastEpoch + 1

        # new to add model graph to tensorboard
        # dataiter = enumerate(loader)
        # ques_id, feats, boxes, sent, target = dataiter.next()
        # writer.add_graph(self.model, (feats, boxes, sent))
        ####################################################
        print('start running epochs')
        for epoch in range(start_epoch, args.epochs):
            print("Start new epoch - epoch number : "+str(epoch))
            quesid2ans = {}
            for i, (ques_id, feats, boxes, sent, target, img_id, img_info) in iter_wrapper(enumerate(loader)):

                self.model.train()

                self.optim.zero_grad()

                feats, boxes, target = feats.cuda(), boxes.cuda(), target.cuda()  # target 32 size

                target_answers = []
                for target_label in (target.max(1)[1]).cpu().numpy():
                    target_ans = dset.label2ans[target_label]
                    target_answers.append(target_ans)

                logit = self.model(feats, boxes, sent,
                                   target_answers, t='qa_woi')

                assert logit.dim() == target.dim() == 2

                loss = self.bce_loss(logit, target)

                loss = loss * logit.size(1)

                loss.backward()

                # /////////////////////////////////////////// #new
                running_loss += loss.item()

                if i % 100 == 99:    # every 1000 mini-batches...

                    # ...log the running loss
                    writer.add_scalar('training loss',
                                      running_loss / 100,
                                      epoch * len(loader) + i)
                    running_loss = 0

                    # ...log the validation loss
                    if self.valid_tuple is not None:
                        valid_score = self.evaluate(eval_tuple)
                        writer.add_scalar('validation loss',
                                          valid_score,
                                          epoch * len(loader) + i)   # x-axis is the number of batches
                # //////////////////////////////////////////

                nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                self.optim.step()

                score, label = logit.max(1)
                for qid, l in zip(ques_id, label.cpu().numpy()):
                    ans = dset.label2ans[l]
                    quesid2ans[qid.item()] = ans
            if(epoch % 10 == 0):
                self.newSave(epoch, running_loss)  # save model when epoch = 50

            log_str = "\nEpoch- %d: Train %0.2f\n" % (
                epoch, evaluator.evaluate(quesid2ans) * 100.)

            if self.valid_tuple is not None:  # Do Validation
                valid_score = self.evaluate(eval_tuple)
                if valid_score > best_valid:
                    best_valid = valid_score
                    print('model checkpoint saved  epoch:'+str(epoch))
                    # self.save("BEST")
                    # self.newSave(epoch, running_loss)

                log_str += "Epoch- %d: Valid %0.2f\n" % (epoch, valid_score * 100.) + \
                           "Epoch- %d: Best %0.2f\n" % (
                               epoch, best_valid * 100.)

            print(log_str, end='')

            with open(self.output + "/log.log", 'a') as f:
                print('logged the epoch result')
                f.write(log_str)
                f.flush()

        writer.flush()  # new
        writer.close()

        self.save("LAST")

    def predict(self, eval_tuple: DataTuple, dump=None):
        """
        Predict the answers to questions in a data split.
        :param eval_tuple: The data tuple to be evaluated.
        :param dump: The path of saved file to dump results.
        :return: A dict of question_id to answer.
        """
        self.model.eval()
        dset, loader, evaluator = eval_tuple
        quesid2ans = {}
        true_count = 0
        false_count = 0
        for i, datum_tuple in enumerate(loader):
            # Avoid seeing ground truth
            ques_id, feats, boxes, sent, target, img_id, img_info = datum_tuple
            with torch.no_grad():
                feats, boxes = feats.cuda(), boxes.cuda()

                target_answers = []
                for target_label in (target.max(1)[1]).cpu().numpy():
                    target_ans = dset.label2ans[target_label]
                    target_answers.append(target_ans)

                logit = self.model(feats, boxes, sent,
                                   target_answers)
                score, label = logit.max(1)
                for qid, l in zip(ques_id, label.cpu().numpy()):
                    ans = dset.label2ans[l]
                    quesid2ans[qid.item()] = ans

                    # log.write(str(i)+'\n')
                for qid, l, sentence, targ, imageId, in zip(ques_id, label.cpu().numpy(), sent.cpu().numpy(), target.cpu().numpy(), img_id.cpu().numpy()):
                    ans = dset.label2ans[l]
                    log_str = "image id : " + str(imageId) + "Question : " + str(
                        sentence) + "Target : " + str(targ) + "Predicted : " + str(ans)
                    if(ans == targ):
                        true_count += 1
                    else:
                        false_count += 1
                    LogFile.write(log_str+'\n')
        total = false_count+true_count
        print("True Count : " + str(true_count) +
              " - " + str(true_count/total))
        print("False Count : " + str(false_count) +
              " - " + str(false_count/total))
        print("Total Count : " + str(total))

        if dump is not None:
            evaluator.dump_result(quesid2ans, dump)
        return quesid2ans

    def evaluate(self, eval_tuple: DataTuple, dump=None):
        """Evaluate all data in data_tuple."""
        quesid2ans = self.predict(eval_tuple, dump)
        return eval_tuple.evaluator.evaluate(quesid2ans)

    @ staticmethod
    def oracle_score(data_tuple):
        dset, loader, evaluator = data_tuple
        quesid2ans = {}
        for i, (ques_id, feats, boxes, sent, target, img_id, img_info) in enumerate(loader):
            _, label = target.max(1)
            for qid, l in zip(ques_id, label.cpu().numpy()):
                ans = dset.label2ans[l]
                quesid2ans[qid.item()] = ans
        return evaluator.evaluate(quesid2ans)

    def save(self, name):
        print(os.path.join(self.output, "%s.pth" % name))
        torch.save(self.model.state_dict(),
                   os.path.join(self.output, "%s.pth" % name))

    def load(self, path):
        print("Load model from %s" % path)
        state_dict = torch.load("%s.pth" % path)
        self.model.load_state_dict(state_dict)

    def newLoadModel(self):
        PATH = adv_model_dir
        checkpoint = torch.load(PATH)
        return checkpoint
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optim.load_state_dict(checkpoint['optimizer_state_dict'])

    def getLastEpoch(self):
        PATH = checkpoint_dir
        checkpoint = torch.load(PATH)
        return checkpoint

    def newSave(self, EPOCH, LOSS):
        # PATH = checkpoint_dir
        PATH = temp_checkpoint_save_dir
        torch.save({
            'epoch': EPOCH,
            'model_lxrt': self.model.lxrt_encoder,
            'model_lxrt_state_dict': self.model.lxrt_encoder.state_dict(),
            'model_state_dict': self.model.state_dict(),
            'full_model': self.model,
            'optimizer_state_dict': self.optim.state_dict(),
            'loss': LOSS,
        }, PATH)


if __name__ == '__main__':

    pvqa = PVQA()
    # print('first')
    # if(startFrom == "M"):
    #     pvqa.newLoadModel()
    # # Load VQA model weights
    # # Note: It is different from loading LXMERT pre-trained weights.
    # if args.load is not None:
    #     # pvqa.load(args.load)
    #     # pvqa.newLoadModel()
    #     print('second')
    # else:
    #     print('third')

    # Test or Train
    if(True):
        args.fast = args.tiny = False  # Always loading all data in test
        if(True):
            result = pvqa.evaluate(
                get_data_tuple('test', bs=valid_bs,
                               shuffle=False, drop_last=False),
                dump=os.path.join(args.output, 'test_predict.json')
            )
            print(result)
            with open(args.output + "/log.log", 'a') as f:
                f.write('test result=' + str(result))
                f.flush()

        elif 'val' in args.test:
            # NOT USED
            # Since part of valididation data are used in pre-training/fine-tuning,
            # only validate on the minival set.
            result = pvqa.evaluate(
                get_data_tuple('test', bs=valid_bs,
                               shuffle=False, drop_last=False),
                dump=os.path.join(args.output, 'test_predict.json')
            )
            print(result)
            with open(args.output + "/log.log", 'a') as f:
                f.write('test result=' + str(result))
                f.flush()

        else:
            assert False, "No such test option for %s" % args.test
    else:
        print('Splits in Train data:', pvqa.train_tuple.dataset.splits)
        if pvqa.valid_tuple is not None:
            print('Splits in Valid data:', pvqa.valid_tuple.dataset.splits)
            print("Valid Oracle: %0.2f" %
                  (pvqa.oracle_score(pvqa.valid_tuple) * 100))
        else:
            print("DO NOT USE VALIDATION")
        pvqa.train(pvqa.train_tuple, pvqa.valid_tuple)
