#
# This file is part of vantgrd-py. vantgrd-py is free software: you can
# redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 2.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this program; if not, write to the Free Software Foundation, Inc., 51
# Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
# Copyright 2017 Davide Anastasia
#
from datetime import datetime


class ClassificationTrainTracker:
    def __init__(self, rate=0):
        self.rate = rate

        self.log_likelihood_ = 0.
        self.loss_ = []

        self.current_sample_ = 0

        self.start_train_time_ = None
        self.end_train_time_ = None

        self.current_epoch_ = 0
        self.start_epoch_time_ = None
        self.end_epoch_time_ = None

    def clear(self):
        self.log_likelihood_ = 0.
        self.loss_ = []

        self.current_sample_ = 0

        self.start_train_time_ = None
        self.end_train_time_ = None

        self.current_epoch_ = 0
        self.start_epoch_time_ = None
        self.end_epoch_time_ = None

    def start_train(self):
        self.start_train_time_ = datetime.now()

    def end_train(self):
        self.end_train_time_ = datetime.now()
        if self.rate > 0:
            print(' --- TRAINING FINISHED IN {0} SECONDS WITH LOSS {1:.2f} ---'.format((self.end_train_time_ - self.start_train_time_).seconds,
                                                                                       self.log_likelihood_/self.current_sample_))

    def start_epoch(self, n_epoch):
        self.current_epoch_ = n_epoch
        self.start_epoch_time_ = datetime.now()
        #
        # if self.rate > 0:
        #     print('TRAINING EPOCH: {0:2}'.format(n_iter + 1))
        #     print('-' * 18)

    def end_epoch(self):
        self.end_epoch_time_ = datetime.now()
        #
        # if self.rate > 0:
        #     print('EPOCH {0:2} FINISHED IN {1} seconds'.format(
        #         n_iter + 1, (datetime.now() - epoch_time).seconds))

    def track(self, log_likelihood):
        self.current_sample_ += 1
        self.log_likelihood_ += log_likelihood
        if self.rate > 0 and self.current_sample_ % self.rate == 0:
            # Append to the loss list.
            self.loss_.append(self.log_likelihood_/self.current_sample_)

            # Print all the current information
            print('Epoch: {0:3} | '
                  'Training Samples: {1:9} | '
                  'Loss: {2:11.2f} | '
                  'LossAdj: {3:8.5f} | '
                  'Time taken: {4:4} seconds'.format(self.current_epoch_,
                                                     self.current_sample_,
                                                     self.log_likelihood_,
                                                     float(self.log_likelihood_) / self.current_sample_,
                                                     (datetime.now() - self.start_epoch_time_).seconds))
