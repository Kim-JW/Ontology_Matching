# coding:utf-8
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
import os
import time
import datetime
import ctypes
import json


class Config(object):
    r"""In this class, we set the configuration parameters, adopt C library for data and memory processing. In the following,
    we train models and test models.
    """

    def __init__(self):
        self.lib = ctypes.cdll.LoadLibrary("./release/Base.so")
        self.lib.sampling.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                      ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]
        self.lib.getHeadBatch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        self.lib.getTailBatch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        self.lib.testHead.argtypes = [ctypes.c_void_p]
        self.lib.testTail.argtypes = [ctypes.c_void_p]
        self.test_flag = False
        self.in_path = "./OpenKE/openke/"
        self.out_path = "./"
        self.bern = 0
        self.hidden_size = 100
        self.ent_size = self.hidden_size
        self.rel_size = self.hidden_size
        self.train_times = 0
        self.margin = 1.0
        self.nbatches = 100
        self.negative_ent = 1
        self.negative_rel = 0
        self.workThreads = 1
        self.alpha = 0.001
        self.lmbda = 0.000
        self.log_on = 1
        self.lr_decay = 0.000
        self.weight_decay = 0.000
        self.exportName = None
        self.importName = None
        self.export_steps = 0
        self.opt_method = "SGD"
        self.optimizer = None

    def init(self):
        self.trainModel = None
        if self.in_path != None:
            self.lib.setInPath(ctypes.create_string_buffer(self.in_path, len(self.in_path) * 2))
            self.lib.setBern(self.bern)
            self.lib.setWorkThreads(self.workThreads)
            self.lib.randReset()
            self.lib.importTrainFiles()
            self.relTotal = self.lib.getRelationTotal()
            self.entTotal = self.lib.getEntityTotal()
            self.trainTotal = self.lib.getTrainTotal()
            self.batch_size = self.lib.getTrainTotal() / self.nbatches
            self.batch_seq_size = self.batch_size * (1 + self.negative_ent + self.negative_rel)
            self.batch_h = np.zeros(self.batch_size * (1 + self.negative_ent + self.negative_rel), dtype=np.int64)
            self.batch_t = np.zeros(self.batch_size * (1 + self.negative_ent + self.negative_rel), dtype=np.int64)
            self.batch_r = np.zeros(self.batch_size * (1 + self.negative_ent + self.negative_rel), dtype=np.int64)
            self.batch_y = np.zeros(self.batch_size * (1 + self.negative_ent + self.negative_rel), dtype=np.float32)
            self.batch_h_addr = self.batch_h.__array_interface__['data'][0]
            self.batch_t_addr = self.batch_t.__array_interface__['data'][0]
            self.batch_r_addr = self.batch_r.__array_interface__['data'][0]
            self.batch_y_addr = self.batch_y.__array_interface__['data'][0]
        if self.test_flag:
            self.lib.importTestFiles()
            self.test_h = np.zeros(self.lib.getEntityTotal(), dtype=np.int64)
            self.test_t = np.zeros(self.lib.getEntityTotal(), dtype=np.int64)
            self.test_r = np.zeros(self.lib.getEntityTotal(), dtype=np.int64)
            self.test_h_addr = self.test_h.__array_interface__['data'][0]
            self.test_t_addr = self.test_t.__array_interface__['data'][0]
            self.test_r_addr = self.test_r.__array_interface__['data'][0]

    def get_ent_total(self):
        r"""This method gets the entity total of knowledge base.
        """
        return self.entTotal

    def get_rel_total(self):
        r""" This method gets the relation total of knowledge base.
        """
        return self.relTotal

    def set_lmbda(self, lmbda):

        self.lmbda = lmbda

    def set_opt_method(self, method):
        r"""This method sets the optimizer for your model.

        Arguments:
            optimizer: ``SGD`` ``Adagrad`` ``Adam`` and ``Adadelta`` can be chosen for optimizing.
        """
        self.opt_method = method

    def set_test_flag(self, flag):
        r"""This method sets whether we test our model.

        Arguments:
            flag (bool): if True, we test the model.

        .. note:: Note that test_flag must  be set **after** all the other configuration parameters are set.
        """
        self.test_flag = flag

    def set_log_on(self, flag):
        r"""This method sets whether to log on the loss value.

        Arguments:
            flag (bool): if True, logs on the loss value when training.
        """
        self.log_on = flag

    def set_alpha(self, alpha):
        r"""This mothod sets the learning rate for gradient descent.

        Arguments:
            alpha (float): the learning rate.
        """
        self.alpha = alpha

    def set_in_path(self, path):
        r"""This method sets the path of benchmark.
        """
        self.in_path = path

    def set_out_files(self, path):
        r"""This method sets where to emport embedding matrix.
        """
        self.out_path = path

    def set_bern(self, bern):
        r"""This method sets the strategy for negative sampling.

        Arguments:
            bern: "bern" or "unif"
        """
        self.bern = bern

    def set_dimension(self, dim):
        r"""This method sets the entity dimension and  relation dimension at the same time.

        Arguments:
            dim (int): the dimension of entity and relation.
        """
        self.hidden_size = dim
        self.ent_size = dim
        self.rel_size = dim

    def set_ent_dimension(self, dim):
        r"""This method sets the dimension of entity.

        Arguments:
            dim (int): the dimension of entity.
        """
        self.ent_size = dim

    def set_rel_dimension(self, dim):
        r"""This method sets the dimension of relation.

        Arguments:
            dim (int): the dimension of relation.
        """
        self.rel_size = dim

    def set_train_times(self, times):
        r"""This method sets the rounds for training.

        Arguments:
            times (int): rounds for training.
        """
        self.train_times = times

    def set_nbatches(self, nbatches):
        r"""This method sets the number of batch.

        Arguments:
            nbatches (int): number of batch.

        """
        self.nbatches = nbatches

    def set_margin(self, margin):
        r"""This method sets the margin for the widely used pairwise margin-based ranking loss.

        Arguments:
            margin (float): margin for margin-based ranking function
        """
        self.margin = margin

    def set_work_threads(self, threads):
        r"""We can use multi-threading trainning for accelaration. This method sets the numebr of threads.

        Arguments:
            threads (int): number of working threads.

        """
        self.workThreads = threads

    def set_ent_neg_rate(self, rate):
        r"""the number of negatives generated per positive training sample influnces the experiment results.
        This method sets the number of negative entities constructed per positive sample.

        Arguments:
            rate (int): the number of negative entities per positive sample.
        """
        self.negative_ent = rate

    def set_rel_neg_rate(self, rate):
        r"""This method sets the number of negative relations per positive sample.

        Arguments:
            rate (int): the number of negative relations per positive sample.
        """
        self.negative_rel = rate

    def set_import_files(self, path):
        r"""Model paramters are exported automatically every few rounds.
        This method sets the path to find exported model parameters.

        Arguments:
            path: path to automatically exported model parameters.
        """
        self.importName = path

    def set_export_files(self, path):
        r"""Model parameters will be exported to this path automatically.

        Arguments:
            path: files that model parameters will be exported to.
        """
        self.exportName = path

    def set_export_steps(self, steps):
        r""" This method sets that every few steps the model paramters will be exported automatically.

        Arguments:
            steps (int): Models will be exported via torch.save() automatically every few rounds

        """
        self.export_steps = steps

    def set_lr_decay(self, lr_decay):
        r"""This method sets the learning rate decay for ``Adagrad`` optim method.

        Arguments:
            lr_decay (float): learning rate decay
        """
        self.lr_decay = lr_decay

    def set_weight_decay(self, weight_decay):
        r"""This method sets the weight decay for ``Adagrad`` optim method.

        Arguments:
            weight_decay (float): weight decay  for ``Adagrad``.
        """
        self.weight_decay = weight_decay

    def sampling(self):
        r"""In this function, we choose positive samples and construct negative samples.
        """
        self.lib.sampling(self.batch_h_addr, self.batch_t_addr, self.batch_r_addr, self.batch_y_addr, self.batch_size,
                          self.negative_ent, self.negative_rel)

    def save_pytorch(self):
        r"""This method saves the model paramters to ``self.exportName`` which was set by :func:`set_export_files`.
        """
        torch.save(self.trainModel.state_dict(), self.exportName)

    def restore_pytorch(self):
        r"""This method restore model through ``torch.load``
        """
        self.trainModel.load_state_dict(torch.load(self.importName))

    def export_variables(self, path=None):
        r"""This method export model paramters through ``torch.save``.

        Arguments:
            path: If None, this function euquals to :func:`save_pytorch`, else save paramters to ``path``

        """
        if path == None:
            torch.save(self.trainModel.state_dict(), self.exportName)
        else:
            torch.save(self.trainModel.state_dict(), path)

    def import_variables(self, path=None):
        r"""This method export model paramters through ``torch.load``.

        Arguments:
            path: If None, this function euquals to :func:`restore_pytorch`, else save paramters to ``path``

        """
        if path == None:
            self.trainModel.load_state_dict(torch.load(self.importName))
        else:
            self.trainModel.load_state_dict(torch.load(path))

    def get_parameter_lists(self):
        return self.trainModel.cpu().state_dict()

    def get_parameters_by_name(self, var_name):
        return self.trainModel.cpu().state_dict().get(var_name)

    def get_parameters(self, mode="numpy"):
        r"""This method gets the model paramters.

        Arguments:
            mode: if ``numpy``, returns model parameters as numpy array, if ``list``, returns those as list

        """
        res = {}
        lists = self.get_parameter_lists()
        for var_name in lists:
            if mode == "numpy":
                res[var_name] = lists[var_name].numpy()
            if mode == "list":
                res[var_name] = lists[var_name].numpy().tolist()
            else:
                res[var_name] = lists[var_name]
        return res



def save_parameters(self, path=None):
    r"""This method save model parameters as json files when training finished.

    Arguments:
        path: if None, save parameters to ``self.out_path`` which was set by :func:`set_out_files`.
    """
    if path == None:
        path = self.out_path
    f = open(path, "w")
    f.write(json.dumps(self.get_parameters("list")))
    f.close()


def set_parameters_by_name(self, var_name, tensor):
    self.trainModel.state_dict().get(var_name).copy_(torch.from_numpy(np.array(tensor)))


def set_parameters(self, lists):
    for i in lists:
        self.set_parameters_by_name(i, lists[i])



def set_model(self, model):
    r"""This method sets the traing model and optimizer method.

    Arguments:
        model: training model. We can choose from :class:``models.TransE`` :class:``models.TransH`` :class:``models.TransR`` :class:``models.TransD`` :class:``models.RESCAL`` :class:``models.DistMult`` and :class:``models.ComplEx``
    """
    self.model = model
    self.trainModel = self.model(config=self)
    self.trainModel.cuda()
    if self.optimizer != None:
        pass
    elif self.opt_method == "Adagrad" or self.opt_method == "adagrad":
        self.optimizer = optim.Adagrad(self.trainModel.parameters(), lr=self.alpha, lr_decay=self.lr_decay,
                                       weight_decay=self.weight_decay)
    elif self.opt_method == "Adadelta" or self.opt_method == "adadelta":
        self.optimizer = optim.Adadelta(self.trainModel.parameters(), lr=self.alpha)
    elif self.opt_method == "Adam" or self.opt_method == "adam":
        self.optimizer = optim.Adam(self.trainModel.parameters(), lr=self.alpha)
    else:
        self.optimizer = optim.SGD(self.trainModel.parameters(), lr=self.alpha)



def run(self):
    r"""In this function, we train the model"""
    if self.importName != None:
        self.restore_pytorch()
    for epoch in range(self.train_times):
        res = 0.0
        for batch in range(self.nbatches):
            self.sampling()
            self.optimizer.zero_grad()
            loss = self.trainModel()
            res = res + loss.data[0]
            loss.backward()
            self.optimizer.step()
        if self.exportName != None and (self.export_steps != 0 and epoch % self.export_steps == 0):
            self.save_pytorch()
        if self.log_on == 1:
            print
            epoch
            print
            res
    if self.exportName != None:
        self.save_pytorch()
    if self.out_path != None:
        self.save_parameters(self.out_path)


def test(self):
    r"""In this function, we test the model."""
    if self.importName != None:
        self.restore_pytorch()
    # self.trainModel.cuda()
    total = self.lib.getTestTotal()
    for epoch in range(total):
        self.lib.getHeadBatch(self.test_h_addr, self.test_t_addr, self.test_r_addr)
        res = self.trainModel.predict(self.test_h, self.test_t, self.test_r)
        self.lib.testHead(res.data.numpy().__array_interface__['data'][0])

        self.lib.getTailBatch(self.test_h_addr, self.test_t_addr, self.test_r_addr)
        res = self.trainModel.predict(self.test_h, self.test_t, self.test_r)
        self.lib.testTail(res.data.numpy().__array_interface__['data'][0])
        if self.log_on:
            print
            epoch
    self.lib.test()
