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
    "/pvqa_path.pth"
LogFileTrue = baseUrl+'/debug_true.txt'
LogFileFalse = baseUrl+'/debug_false.txt'
# f = open(os.path.dirname(os.path.abspath(__file__)+"/"+writeFileName, "w+")
f = open(LogFileTrue, "w+")
f = open(LogFileFalse, "w+")
# f.close()
LogFileTrue = open(LogFileTrue, "r+")
LogFileFalse = open(LogFileFalse, "r+")

# feats logger
FeatureLogFilePath = baseUrl+'/features.txt'
f = open(FeatureLogFilePath, "w+")
# f.close()

#FeatureLogFile = open(FeatureLogFilePath, "r+")


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
                # print("target_answers " + str(target_answers))
                # break
                logit = self.model(feats, boxes, sent,
                                   target_answers)

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
        tList = ['test_0001', 'test_0005', 'test_0006', 'test_0009', 'test_0011', 'test_0012', 'test_0013', 'test_0017', 'test_0018', 'test_0019', 'test_0021', 'test_0022', 'test_0023', 'test_0024', 'test_0025', 'test_0026', 'test_0031', 'test_0032', 'test_0035', 'test_0036', 'test_0037', 'test_0038', 'test_0040', 'test_0041', 'test_0042', 'test_0043', 'test_0044', 'test_0045', 'test_0048', 'test_0050', 'test_0055', 'test_0056', 'test_0057', 'test_0060', 'test_0061', 'test_0062', 'test_0063', 'test_0065', 'test_0067', 'test_0068', 'test_0069', 'test_0070', 'test_0071', 'test_0072', 'test_0073', 'test_0076', 'test_0078', 'test_0080', 'test_0081', 'test_0084', 'test_0085', 'test_0087', 'test_0088', 'test_0089', 'test_0093', 'test_0098', 'test_0102', 'test_0104', 'test_0108', 'test_0110', 'test_0111', 'test_0112', 'test_0113', 'test_0115', 'test_0116', 'test_0120', 'test_0122', 'test_0123', 'test_0124', 'test_0128', 'test_0129', 'test_0130', 'test_0131', 'test_0133', 'test_0137', 'test_0138', 'test_0140', 'test_0142', 'test_0143', 'test_0144', 'test_0148', 'test_0151', 'test_0152', 'test_0153', 'test_0154', 'test_0157', 'test_0159', 'test_0162', 'test_0163', 'test_0165', 'test_0166', 'test_0168', 'test_0169', 'test_0173', 'test_0177', 'test_0180', 'test_0183', 'test_0184', 'test_0187', 'test_0188', 'test_0189', 'test_0190', 'test_0191', 'test_0193', 'test_0194', 'test_0195', 'test_0198', 'test_0199', 'test_0206', 'test_0209', 'test_0210', 'test_0211', 'test_0212', 'test_0216', 'test_0217', 'test_0218', 'test_0219', 'test_0220', 'test_0225', 'test_0226', 'test_0228', 'test_0232', 'test_0233', 'test_0234', 'test_0235', 'test_0236', 'test_0237', 'test_0238', 'test_0239', 'test_0241', 'test_0245', 'test_0246', 'test_0247', 'test_0249', 'test_0252', 'test_0253', 'test_0254', 'test_0255', 'test_0256', 'test_0259', 'test_0264', 'test_0265', 'test_0272', 'test_0274', 'test_0275', 'test_0278', 'test_0279', 'test_0280', 'test_0282', 'test_0284', 'test_0285', 'test_0288', 'test_0291', 'test_0294', 'test_0295', 'test_0297', 'test_0298', 'test_0299', 'test_0304', 'test_0310', 'test_0312', 'test_0313', 'test_0315', 'test_0317', 'test_0318', 'test_0319', 'test_0320', 'test_0321', 'test_0322', 'test_0323', 'test_0324', 'test_0325', 'test_0326', 'test_0327', 'test_0328', 'test_0329', 'test_0330', 'test_0331', 'test_0332', 'test_0333', 'test_0334', 'test_0335', 'test_0336', 'test_0337', 'test_0338', 'test_0339', 'test_0340', 'test_0341', 'test_0342', 'test_0343', 'test_0344', 'test_0345', 'test_0346', 'test_0347', 'test_0348', 'test_0349', 'test_0350', 'test_0351', 'test_0352', 'test_0353', 'test_0354', 'test_0355', 'test_0356', 'test_0357', 'test_0358', 'test_0359', 'test_0360', 'test_0361', 'test_0362', 'test_0363', 'test_0364', 'test_0365', 'test_0366', 'test_0367', 'test_0368', 'test_0369', 'test_0370', 'test_0371', 'test_0372', 'test_0373', 'test_0374', 'test_0393', 'test_0394', 'test_0395', 'test_0396', 'test_0397', 'test_0398', 'test_0399', 'test_0400', 'test_0401', 'test_0402', 'test_0403', 'test_0404', 'test_0405', 'test_0406', 'test_0407', 'test_0408', 'test_0409', 'test_0410',
                 'test_0436', 'test_0462', 'test_0463', 'test_0464', 'test_0465', 'test_0466', 'test_0467', 'test_0468', 'test_0469', 'test_0470', 'test_0471', 'test_0472', 'test_0473', 'test_0474', 'test_0484', 'test_0485', 'test_0486', 'test_0487', 'test_0488', 'test_0489', 'test_0490', 'test_0491', 'test_0492', 'test_0493', 'test_0494', 'test_0495', 'test_0513', 'test_0514', 'test_0515', 'test_0516', 'test_0517', 'test_0518', 'test_0519', 'test_0520', 'test_0521', 'test_0522', 'test_0523', 'test_0524', 'test_0525', 'test_0526', 'test_0527', 'test_0528', 'test_0529', 'test_0530', 'test_0531', 'test_0532', 'test_0533', 'test_0534', 'test_0535', 'test_0536', 'test_0537', 'test_0561', 'test_0562', 'test_0563', 'test_0564', 'test_0565', 'test_0566', 'test_0567', 'test_0568', 'test_0569', 'test_0570', 'test_0571', 'test_0572', 'test_0573', 'test_0574', 'test_0575', 'test_0576', 'test_0577', 'test_0578', 'test_0579', 'test_0580', 'test_0581', 'test_0582', 'test_0606', 'test_0607', 'test_0608', 'test_0609', 'test_0610', 'test_0631', 'test_0632', 'test_0633', 'test_0634', 'test_0635', 'test_0636', 'test_0637', 'test_0638', 'test_0639', 'test_0640', 'test_0641', 'test_0642', 'test_0643', 'test_0644', 'test_0645', 'test_0646', 'test_0647', 'test_0648', 'test_0649', 'test_0650', 'test_0651', 'test_0652', 'test_0653', 'test_0654', 'test_0655', 'test_0692', 'test_0693', 'test_0694', 'test_0695', 'test_0696', 'test_0697', 'test_0698', 'test_0699', 'test_0700', 'test_0701', 'test_0702', 'test_0703', 'test_0704', 'test_0705', 'test_0706', 'test_0707', 'test_0724', 'test_0725', 'test_0726', 'test_0727', 'test_0728', 'test_0729', 'test_0730', 'test_0731', 'test_0732', 'test_0733', 'test_0734', 'test_0735', 'test_0736', 'test_0737', 'test_0738', 'test_0739', 'test_0740', 'test_0741', 'test_0742', 'test_0743', 'test_0744', 'test_0745', 'test_0746', 'test_0747', 'test_0748', 'test_0749', 'test_0750', 'test_0751', 'test_0752', 'test_0753', 'test_0754', 'test_0755', 'test_0756', 'test_0757', 'test_0758', 'test_0759', 'test_0760', 'test_0761', 'test_0762', 'test_0763', 'test_0764', 'test_0765', 'test_0766', 'test_0767', 'test_0768', 'test_0805', 'test_0812', 'test_0813', 'test_0814', 'test_0815', 'test_0816', 'test_0817', 'test_0818', 'test_0819', 'test_0820', 'test_0821', 'test_0822', 'test_0823', 'test_0824', 'test_0825', 'test_0826', 'test_0827', 'test_0828', 'test_0887', 'test_0888', 'test_0889', 'test_0890', 'test_0891', 'test_0892', 'test_0893', 'test_0894', 'test_0895', 'test_0896', 'test_0897', 'test_0898', 'test_0899', 'test_0900', 'test_0901', 'test_0902', 'test_0903', 'test_0950', 'test_0951', 'test_0952', 'test_0953', 'test_0954', 'test_0955', 'test_0956', 'test_0957', 'test_0958', 'test_0959', 'test_0960', 'test_0961', 'test_0962', 'test_0963', 'test_0964', 'test_0965', 'test_0966', 'test_0967', 'test_0968', 'test_0969', 'test_0970', 'test_0971', 'test_0972', 'test_0973', 'test_0974', 'test_0975', 'test_0976', 'test_0977', 'test_0978', 'test_0979', 'test_0980', 'test_0981', 'test_0982', 'test_0983', 'test_0984', 'test_0985', 'test_0986', 'test_0987', 'test_0988', 'test_0989', 'test_0990']
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
        noAnsCount = 0
        for i, datum_tuple in enumerate(loader):
            # Avoid seeing ground truth
            ques_id, feats, boxes, sent, target, img_id, img_info = datum_tuple
            with torch.no_grad():
                feats, boxes = feats.cuda(), boxes.cuda()

                target_answers = []
                for i in range(len(img_id)):
                    t = target[i]
                    t = t.tolist()
                    s = max(t)
                    if(int(s) == 0):
                        noAnsCount += 1
                        target_answers.append("Answer out of scope")
                    else:
                        target_label = t.index(s)
                        target_ans = dset.label2ans[target_label]
                        target_answers.append(target_ans)

                # for target_label in (target.max(1)[1]).cpu().numpy():
                #     target_ans = dset.label2ans[target_label]
                #     target_answers.append(target_ans)
                # for i in range(len(img_id)):
                #     if(img_id[i] in tList):
                #         x = feats.tolist()
                #         y = boxes.tolist()
                #         FeatureLogFile.write(str("feats")+'\n')
                #         FeatureLogFile.write(str(x[i])+'\n')
                #         FeatureLogFile.write(str("boxes")+'\n')
                #         FeatureLogFile.write(str(y[i])+'\n')
                # FeatureLogFile.close()

                logit = self.model(feats, boxes, sent,
                                   target_answers)
                score, label = logit.max(1)
                for qid, l in zip(ques_id, label.cpu().numpy()):
                    ans = dset.label2ans[l]
                    quesid2ans[qid.item()] = ans

                for qid, l, sentence, targ, imageId, in zip(ques_id, label.cpu().numpy(), sent, target_answers, img_id):
                    if(imageId in tList):
                        ans = dset.label2ans[l]
                        log_str = "image id : " + str(imageId) + " --- Question : " + str(
                            sentence) + " --- Target : " + str(targ) + " --- Predicted : " + str(ans) + " --- Predicted Label : " + str(l)
                        if(ans == targ):
                            LogFileTrue.write(log_str+'\n')
                            true_count += 1
                        else:
                            LogFileFalse.write(log_str+'\n')
                            false_count += 1
        print("no Ans Count "+str(noAnsCount))
        total = false_count+true_count
        print("True Count : " + str(true_count) +
              " - " + str(round(true_count/total, 2))+"%")
        print("False Count : " + str(false_count) +
              " - " + str(round(false_count/total, 2))+"%")
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
