## Portions of Code from, copyright 2018 Jochen Gast

from __future__ import absolute_import, division, print_function

import numpy as np
import colorama
import logging
import flowbias.logger as logger
import flowbias.tools as tools
import collections
import itertools

import scipy.misc
import torch
import torch.nn as nn
import os
import math

# for evaluation
from flowbias.utils.flow import flow_to_png, flow_to_png_middlebury
from flowbias.utils.flow import write_flow, write_flow_png

# --------------------------------------------------------------------------------
# Exponential moving average smoothing factor for speed estimates
# Ranges from 0 (average speed) to 1 (current/instantaneous speed) [default: 0.3].
# --------------------------------------------------------------------------------
TQDM_SMOOTHING = 0


# -------------------------------------------------------------------------------------------
# Magic progressbar for inputs of type 'iterable'
# -------------------------------------------------------------------------------------------
def create_progressbar(iterable,
                       total=None,
                       desc="",
                       train=False,
                       unit="it",
                       initial=0,
                       offset=0,
                       invert_iterations=False,
                       logging_on_update=False,
                       logging_on_close=True,
                       postfix=False):

    # ---------------------------------------------------------------
    # Pick colors
    # ---------------------------------------------------------------
    reset = colorama.Style.RESET_ALL
    bright = colorama.Style.BRIGHT
    cyan = colorama.Fore.CYAN
    dim = colorama.Style.DIM
    green = colorama.Fore.GREEN

    # ---------------------------------------------------------------
    # Specify progressbar layout:
    #   l_bar, bar, r_bar, n, n_fmt, total, total_fmt, percentage,
    #   rate, rate_fmt, rate_noinv, rate_noinv_fmt, rate_inv,
    #   rate_inv_fmt, elapsed, remaining, desc, postfix.
    # ---------------------------------------------------------------
    bar_format = ""
    bar_format += "%s==>%s%s {desc}:%s " % (cyan, reset, bright, reset)     # description
    bar_format += "{percentage:3.0f}%"                                      # percentage
    bar_format += "%s|{bar}|%s " % (dim, reset)                             # bar
    bar_format += " {n_fmt}/{total_fmt}  "                                  # i/n counter
    bar_format += "{elapsed}<{remaining}"                                   # eta
    if invert_iterations:
        bar_format += " {rate_inv_fmt}  "                                   # iteration timings
    else:
        bar_format += " {rate_noinv_fmt}  "
    bar_format += "%s{postfix}%s" % (green, reset)                          # postfix

    # ---------------------------------------------------------------
    # Specify TQDM arguments
    # ---------------------------------------------------------------
    tqdm_args = {
        "iterable": iterable,
        "desc": desc,                          # Prefix for the progress bar
        "total": len(iterable) if total is None else total,  # The number of expected iterations
        "leave": True,                         # Leave progress bar when done
        "miniters": 1 if train else None,      # Minimum display update interval in iterations
        "unit": unit,                          # String be used to define the unit of each iteration
        "initial": initial,                    # The initial counter value.
        "dynamic_ncols": True,                 # Allow window resizes
        "smoothing": TQDM_SMOOTHING,           # Moving average smoothing factor for speed estimates
        "bar_format": bar_format,              # Specify a custom bar string formatting
        "position": offset,                    # Specify vertical line offset
        "ascii": True,
        "logging_on_update": logging_on_update,
        "logging_on_close": logging_on_close
    }

    return tools.tqdm_with_logging(**tqdm_args)


def tensor2float_dict(tensor_dict):
    return {key: tensor.item() for key, tensor in tensor_dict.items()}


def format_moving_averages_as_progress_dict(moving_averages_dict={},
                                            moving_averages_postfix="avg"):
    progress_dict = collections.OrderedDict([
        (key + moving_averages_postfix, "%1.4f" % moving_averages_dict[key].mean())
        for key in sorted(moving_averages_dict.keys())
    ])
    return progress_dict


def format_learning_rate(lr):
    if np.isscalar(lr):
        return "{}".format(lr)
    else:
        return "{}".format(str(lr[0]) if len(lr) == 1 else lr)


class TrainingEpoch:
    def __init__(self,
                 args,
                 model_and_loss,
                 loader,
                 optimizer,
                 augmentation=None,
                 training_gradient_adjust=None,
                 iters_per_epoch=None,
                 add_progress_stats={},
                 desc="Training Epoch"):

        self._args = args
        self._desc = desc
        self._loader = loader
        self._model_and_loss = model_and_loss
        self._optimizer = optimizer
        self._augmentation = augmentation
        self._training_gradient_adjust = training_gradient_adjust
        self._add_progress_stats = add_progress_stats
        self._iters_per_epoch = iters_per_epoch

    def _step(self, example_dict):

        # -------------------------------------------------------------
        # Get input and target tensor keys
        # -------------------------------------------------------------
        virtual_step = ("virtual_batch" in example_dict) and (example_dict["virtual_batch"] is True)
        if virtual_step:
            input_keys = set()
            target_keys = set()
            for virtual_batch in example_dict["virtual_batches"]:
                input_keys.update(list(filter(lambda x: "input" in x, virtual_batch.keys())))
                target_keys.update(list(filter(lambda x: "target" in x, virtual_batch.keys())))
            tensor_keys = list(input_keys) + list(target_keys)
        else:
            input_keys = list(filter(lambda x: "input" in x, example_dict.keys()))
            target_keys = list(filter(lambda x: "target" in x, example_dict.keys()))
            tensor_keys = input_keys + target_keys

        # -------------------------------------------------------------
        # Possibly transfer to Cuda
        # -------------------------------------------------------------
        if self._args.cuda:
            if virtual_step:
                for vbatch in example_dict["virtual_batches"]:
                    for key, value in vbatch.items():
                        if key in tensor_keys:
                            vbatch[key] = value.cuda(non_blocking=False)
            else:
                for key, value in example_dict.items():
                    if key in tensor_keys:
                        example_dict[key] = value.cuda(non_blocking=False)

        # -------------------------------------------------------------
        # Optionally perform augmentations
        # -------------------------------------------------------------
        if self._augmentation is not None:
            with torch.no_grad():
                if virtual_step:
                    # augment virtual steps
                    for i, vbatch in enumerate(example_dict["virtual_batches"]):
                        example_dict["virtual_batches"][i] = self._augmentation(vbatch)
                else:
                    example_dict = self._augmentation(example_dict)

        # -------------------------------------------------------------
        # Convert inputs/targets to variables that require gradients
        # -------------------------------------------------------------
        if virtual_step:
            for vbatch in example_dict["virtual_batches"]:
                for key, tensor in example_dict.items():
                    if key in input_keys:
                        vbatch[key] = tensor.requires_grad_(True)
                    elif key in target_keys:
                        vbatch[key] = tensor.requires_grad_(False)
        else:
            for key, tensor in example_dict.items():
                if key in input_keys:
                    example_dict[key] = tensor.requires_grad_(True)
                elif key in target_keys:
                    example_dict[key] = tensor.requires_grad_(False)

        # -------------------------------------------------------------
        # Extract batch size from first input
        # -------------------------------------------------------------
        if virtual_step:
            vbatch = example_dict["virtual_batches"][0]
            example_input_key = next(input_key for input_key in vbatch.keys() if "input" in input_key)
            batch_size = vbatch[example_input_key].size()[0]
        else:
            example_input_key = next(input_key for input_key in example_dict.keys() if "input" in input_key)
            batch_size = example_dict[example_input_key].size()[0]

        # -------------------------------------------------------------
        # Reset gradients
        # -------------------------------------------------------------
        self._optimizer.zero_grad()

        # -------------------------------------------------------------
        # Run forward pass to get losses and outputs.
        # -------------------------------------------------------------
        if virtual_step:
            training_loss = None
            if len(example_dict["virtual_batches"]) == 0:
                raise RuntimeError("dict contains no virtual batches")
            # perform virtual steps
            for i, vbatch in enumerate(example_dict["virtual_batches"]):
                vbatch_dict = dict(example_dict)
                vbatch_dict["_vbatch_id"] = i
                for key, value in vbatch.items():
                    if key in vbatch_dict:
                        raise RuntimeError("key {key} of vbatch already exists in example dict")
                    vbatch_dict[key] = value
                loss_dict, output_dict = self._model_and_loss(vbatch_dict)

                # -------------------------------------------------------------
                # Check total_loss for NaNs
                # -------------------------------------------------------------
                training_loss_vb = loss_dict[self._args.training_key]
                assert (not np.isnan(training_loss_vb.item())), "training_loss is NaN"

                if training_loss is None:
                    training_loss = training_loss_vb
                else:
                    training_loss += training_loss_vb
        else:
            loss_dict, output_dict = self._model_and_loss(example_dict)

            # -------------------------------------------------------------
            # Check total_loss for NaNs
            # -------------------------------------------------------------
            training_loss = loss_dict[self._args.training_key]
            assert (not np.isnan(training_loss.item())), "training_loss is NaN"

        # -------------------------------------------------------------
        # Back propagation
        # -------------------------------------------------------------
        if self._training_gradient_adjust is not None:
            self._training_gradient_adjust.adjust_gradients(self._model_and_loss.model)
        training_loss.backward()
        self._optimizer.step()

        # -------------------------------------------------------------
        # Return success flag, loss and output dictionary
        # -------------------------------------------------------------
        return loss_dict, output_dict, batch_size, virtual_step

    def run(self, offset=0):
        # ---------------------------------------
        # Tell model that we want to train
        # ---------------------------------------
        self._model_and_loss.train()

        # ---------------------------------------
        # Keep track of moving averages
        # ---------------------------------------
        moving_averages_dict = None

        #
        #
        #
        if self._iters_per_epoch is None:
            iterator = self._loader
            total = None
        else:
            # if the dataset loops infinitely, we have to restrict the
            # iterations per epoch
            # -> wrap the dataloader
            iterator = itertools.islice(self._loader, self._iters_per_epoch)
            #iterator = self._loader
            total = self._iters_per_epoch


        # ---------------------------------------
        # Progress bar arguments
        # ---------------------------------------
        progressbar_args = {
            "iterable": iterator,
            "total": total,
            "desc": self._desc,
            "train": True,
            "offset": offset,
            "logging_on_update": False,
            "logging_on_close": True,
            "postfix": True
        }

        # ---------------------------------------
        # Perform training steps
        # ---------------------------------------
        with create_progressbar(**progressbar_args) as progress:
            for example_dict in progress:
                loss_dict_per_step, output_dict, batch_size, is_virtual_step = self._step(example_dict)

                # convert
                loss_dict_per_step = tensor2float_dict(loss_dict_per_step)

                # --------------------------------------------------------
                # Possibly initialize moving averages
                # --------------------------------------------------------
                if moving_averages_dict is None:
                    moving_averages_dict = {
                        key: tools.MovingAverage() for key in loss_dict_per_step.keys()
                    }

                # --------------------------------------------------------
                # Add moving averages
                # --------------------------------------------------------
                for key, loss in loss_dict_per_step.items():
                    moving_averages_dict[key].add_average(loss, addcount=batch_size)

                # view statistics in progress bar
                progress_stats = format_moving_averages_as_progress_dict(
                    moving_averages_dict=moving_averages_dict,
                    moving_averages_postfix="_ema_vbatch" if is_virtual_step else "_ema")

                progress.set_postfix(progress_stats)

        # -------------------------------------------------------------
        # Return loss and output dictionary
        # -------------------------------------------------------------
        ema_loss_dict = { key: ma.mean() for key, ma in moving_averages_dict.items() }
        return ema_loss_dict


class EvaluationEpoch:
    def __init__(self,
                 args,
                 model_and_loss,
                 loader,
                 augmentation=None,
                 add_progress_stats={},
                 desc="Evaluation Epoch"):
        self._args = args
        self._desc = desc
        self._loader = loader
        self._model_and_loss = model_and_loss
        self._add_progress_stats = add_progress_stats
        self._augmentation = augmentation
        self._save_output = False
        if self._args.save_result_img or self._args.save_result_flo or self._args.save_result_png:
            self._save_output = True

    def save_outputs(self, example_dict, output_dict):

        # save occ
        save_root_img = self._args.save + '/img/'
        save_root_flo = self._args.save + '/flo/'

        if self._args.save_result_bidirection:
            flow_f = output_dict["flow"].data.cpu().numpy()
            flow_b = output_dict["flow_b"].data.cpu().numpy()
            b_size = output_dict["flow"].data.size(0)
        else:
            flow_f = output_dict["flow"].data.cpu().numpy()
            b_size = output_dict["flow"].data.size(0)

        if self._args.save_result_occ:
            if self._args.save_result_bidirection:
                output_occ = np.round(
                    nn.Sigmoid()(output_dict["occ"]).expand(-1, 3, -1, -1).data.cpu().numpy().transpose(
                        [0, 2, 3, 1])) * 255
                output_occ_b = np.round(
                    nn.Sigmoid()(output_dict["occ_b"]).expand(-1, 3, -1, -1).data.cpu().numpy().transpose(
                        [0, 2, 3, 1])) * 255
            else:
                output_occ = np.round(
                    nn.Sigmoid()(output_dict["occ"]).expand(-1, 3, -1, -1).data.cpu().numpy().transpose(
                        [0, 2, 3, 1])) * 255

        # file names
        file_names_img = []
        file_names_flo = []
        for ii in range(0, b_size):
            if "basedir" in  example_dict.keys():
                file_name_img = save_root_img + example_dict["basedir"][ii] + '/' + str(example_dict["basename"][ii])
                file_name_flo = save_root_flo + example_dict["basedir"][ii] + '/' + str(example_dict["basename"][ii])
                file_names_img.append(file_name_img)
                file_names_flo.append(file_name_flo)
            else:
                file_name_img = save_root_img + '/' + str(example_dict["basename"][ii])
                file_name_flo = save_root_flo + '/' + str(example_dict["basename"][ii])
                file_names_img.append(file_name_img)
                file_names_flo.append(file_name_flo)

            directory_img = os.path.dirname(file_name_img)
            if not os.path.exists(directory_img):
                os.makedirs(directory_img)
            directory_flo = os.path.dirname(file_name_flo)
            if not os.path.exists(directory_flo):
                os.makedirs(directory_flo)

        if self._args.save_result_img:
            for ii in range(0, b_size):
                if self._args.save_result_occ:
                    file_name_occ = file_names_img[ii] + '_occ.png'
                    scipy.misc.imsave(file_name_occ, output_occ[ii])

                    if self._args.save_result_bidirection:
                        scipy.misc.imsave(file_names_img[ii] + '_occ_b.png', output_occ_b[ii])

                # flow vis
                flow_f_rgb = flow_to_png_middlebury(flow_f[ii, ...])
                file_name_flo_vis = file_names_img[ii] + '_flow.png'
                scipy.misc.imsave(file_name_flo_vis, flow_f_rgb)

                if self._args.save_result_bidirection:
                    flow_b_rgb = flow_to_png_middlebury(flow_b[ii, ...])
                    file_name_flo_vis = file_names_img[ii] + '_flow_b.png'
                    scipy.misc.imsave(file_name_flo_vis, flow_b_rgb)

        if self._args.save_result_flo or self._args.save_result_png:
            for ii in range(0, b_size):
                if self._args.save_result_flo:
                    file_name = file_names_flo[ii] + '.flo'
                    write_flow(file_name, flow_f[ii, ...].swapaxes(0, 1).swapaxes(1, 2))
                if self._args.save_result_png:
                    file_name = file_names_flo[ii] + '.png'
                    write_flow_png(file_name, flow_f[ii, ...].swapaxes(0, 1).swapaxes(1, 2))


    def _step(self, example_dict):
        # -------------------------------------------------------------
        # Get input and target tensor keys
        # -------------------------------------------------------------
        input_keys = list(filter(lambda x: "input" in x, example_dict.keys()))
        target_keys = list(filter(lambda x: "target" in x, example_dict.keys()))
        tensor_keys = input_keys + target_keys

        # -------------------------------------------------------------
        # Possibly transfer to Cuda
        # -------------------------------------------------------------
        if self._args.cuda:
            for key, value in example_dict.items():
                if key in tensor_keys:
                    example_dict[key] = value.cuda(non_blocking=False)

        # -------------------------------------------------------------
        # Optionally perform augmentations
        # -------------------------------------------------------------
        if self._augmentation is not None:
            example_dict = self._augmentation(example_dict)

        # -------------------------------------------------------------
        # Extract batch size from first input
        # -------------------------------------------------------------
        example_input_key = next(input_key for input_key in example_dict.keys() if "input" in input_key)
        batch_size = example_dict[example_input_key].size()[0]

        # -------------------------------------------------------------
        # Run forward pass to get losses and outputs.
        # -------------------------------------------------------------
        loss_dict, output_dict = self._model_and_loss(example_dict)

        # -------------------------------------------------------------
        # Return loss and output dictionary
        # -------------------------------------------------------------
        return loss_dict, output_dict, batch_size

    def run(self, offset=0):

        with torch.no_grad():

            # ---------------------------------------
            # Tell model that we want to evaluate
            # ---------------------------------------
            self._model_and_loss.eval()

            # ---------------------------------------
            # Keep track of moving averages
            # ---------------------------------------
            moving_averages_dict = None

            # ---------------------------------------
            # Progress bar arguments
            # ---------------------------------------
            progressbar_args = {
                "iterable": self._loader,
                "desc": self._desc,
                "train": False,
                "offset": offset,
                "logging_on_update": False,
                "logging_on_close": True,
                "postfix": True
            }

            # ---------------------------------------
            # Perform evaluation steps
            # ---------------------------------------
            with create_progressbar(**progressbar_args) as progress:
                for example_dict in progress:

                    # ---------------------------------------
                    # Perform forward evaluation step
                    # ---------------------------------------
                    loss_dict_per_step, output_dict, batch_size = self._step(example_dict)

                    # --------------------------------------------------------
                    # Save results
                    # --------------------------------------------------------
                    if self._save_output:
                        self.save_outputs(example_dict, output_dict)

                    # ---------------------------------------
                    # Convert loss dictionary to float
                    # ---------------------------------------
                    loss_dict_per_step = tensor2float_dict(loss_dict_per_step)

                    # --------------------------------------------------------
                    # Possibly initialize moving averages
                    # --------------------------------------------------------
                    if moving_averages_dict is None:
                        moving_averages_dict = {
                            key: tools.MovingAverage() for key in loss_dict_per_step.keys()
                        }

                    # --------------------------------------------------------
                    # Add moving averages
                    # --------------------------------------------------------
                    for key, loss in loss_dict_per_step.items():
                        moving_averages_dict[key].add_average(loss, addcount=batch_size)

                    # view statistics in progress bar
                    progress_stats = format_moving_averages_as_progress_dict(
                        moving_averages_dict=moving_averages_dict,
                        moving_averages_postfix="_avg")

                    progress.set_postfix(progress_stats)

            # -------------------------------------------------------------
            # Record average losses
            # -------------------------------------------------------------
            avg_loss_dict = { key: ma.mean() for key, ma in moving_averages_dict.items() }

            # -------------------------------------------------------------
            # Return average losses and output dictionary
            # -------------------------------------------------------------
            return avg_loss_dict


def exec_runtime(args,
                 checkpoint_saver,
                 model_and_loss,
                 optimizer,
                 lr_scheduler,
                 train_loader,
                 validation_loader,
                 inference_loader,
                 training_augmentation,
                 training_gradient_adjust,
                 validation_augmentation):
    filename_len = int(math.log10(args.total_epochs) + 1)

    # ----------------------------------------------------------------------------------------------
    # Validation schedulers are a bit special:
    # They want to be called with a validation loss..
    # ----------------------------------------------------------------------------------------------
    validation_scheduler = (lr_scheduler is not None and args.lr_scheduler == "ReduceLROnPlateau")

    # --------------------------------------------------------
    # Log some runtime info
    # --------------------------------------------------------
    with logger.LoggingBlock("Runtime", emph=True):
        logging.info("start_epoch: %i" % args.start_epoch)
        logging.info("total_epochs: %i" % args.total_epochs)

    # ---------------------------------------
    # Total progress bar arguments
    # ---------------------------------------
    progressbar_args = {
        "desc": "Progress",
        "initial": args.start_epoch - 1,
        "invert_iterations": True,
        "iterable": range(1, args.total_epochs + 1),
        "logging_on_close": True,
        "logging_on_update": True,
        "postfix": False,
        "unit": "ep"
    }

    # --------------------------------------------------------
    # Total progress bar
    # --------------------------------------------------------
    print(''), logging.logbook('')
    total_progress = create_progressbar(**progressbar_args)
    print("\n")

    # --------------------------------------------------------
    # Remember validation loss
    # --------------------------------------------------------
    if "validation_key_minimize" in args:
        run_validation = True
        best_validation_loss = float("inf") if args.validation_key_minimize else -float("inf")
    else:
        run_validation = False

    for epoch in range(args.start_epoch, args.total_epochs + 1):
        with logger.LoggingBlock("Epoch %i/%i" % (epoch, args.total_epochs), emph=True):
            store_as_best = False

            # --------------------------------------------------------
            # Always report learning rate
            # --------------------------------------------------------
            if lr_scheduler is None:
                logging.info("lr: %s" % format_learning_rate(args.optimizer_lr))
            else:
                if not validation_scheduler:
                    logging.info("lr: %s" % format_learning_rate(lr_scheduler.get_last_lr()))
                else:
                    # ReduceLROnPlateau has no get_lr() function - read lr directly from optimizer ...
                    logging.info("lr: %s" % format_learning_rate([param_group['lr'] for param_group in optimizer.param_groups]))

            # -------------------------------------------
            # Create and run a training epoch
            # -------------------------------------------
            if train_loader is not None:
                avg_loss_dict = TrainingEpoch(
                    args,
                    desc="   Train",
                    model_and_loss=model_and_loss,
                    optimizer=optimizer,
                    loader=train_loader,
                    augmentation=training_augmentation,
                    training_gradient_adjust=training_gradient_adjust,
                    iters_per_epoch=args.training_iters_per_epoch).run()

            # -------------------------------------------
            # Create and run a validation epoch
            # -------------------------------------------
            eval_this_epoch = (args.eval_every_nth is None) or (epoch % args.eval_every_nth == 0)
            if (validation_loader is not None) and eval_this_epoch:
                # ---------------------------------------------------
                # Construct holistic recorder for epoch
                # ---------------------------------------------------
                avg_loss_dict = EvaluationEpoch(
                    args,
                    desc="Validate",
                    model_and_loss=model_and_loss,
                    loader=validation_loader,
                    augmentation=validation_augmentation).run()

                # ----------------------------------------------------------------
                # Evaluate whether this is the best validation_loss
                # ----------------------------------------------------------------
                if run_validation:
                    validation_loss = avg_loss_dict[args.validation_key]
                    if args.validation_key_minimize:
                        store_as_best = validation_loss < best_validation_loss
                    else:
                        store_as_best = validation_loss > best_validation_loss
                    if store_as_best:
                        best_validation_loss = validation_loss

                # --------------------------------------------------------
                # Update standard learning scheduler
                # or Update validation scheduler, if one is in place
                # --------------------------------------------------------
                if lr_scheduler is not None:
                    if not validation_scheduler:
                        lr_scheduler.step()
                    else:
                        lr_scheduler.step(validation_loss, epoch=epoch)

            # ----------------------------------------------------------------
            # Also show best loss on total_progress
            # ----------------------------------------------------------------
            if run_validation:
                total_progress_stats = {
                    "best_" + args.validation_key + "_avg": "%1.4f" % best_validation_loss
                }
            else:
                total_progress_stats = {
                    "validation": "skipped"
                }
            total_progress.set_postfix(total_progress_stats)


            # ----------------------------------------------------------------
            # Bump total progress
            # ----------------------------------------------------------------
            total_progress.update()
            print('')

            # ----------------------------------------------------------------
            # Store checkpoint
            # ----------------------------------------------------------------
            if checkpoint_saver is not None:
                checkpoint_saver.save_latest(
                    directory=args.save,
                    model_and_loss=model_and_loss,
                    stats_dict=dict(avg_loss_dict, epoch=epoch),
                    store_as_best=store_as_best)

                if (args.save_every_nth_checkpoint is not None) and (epoch % args.save_every_nth_checkpoint == 0):
                    checkpoint_saver.save_custom(
                        directory=args.save,
                        model_and_loss=model_and_loss,
                        stats_dict=dict(avg_loss_dict, epoch=epoch),
                        custom_postfix=f"_iter_{epoch:0{filename_len}}"
                    )

            # ----------------------------------------------------------------
            # Vertical space between epochs
            # ----------------------------------------------------------------
            print(''), logging.logbook('')

    # ----------------------------------------------------------------
    # Finish
    # ----------------------------------------------------------------
    total_progress.close()
    logging.info("Finished.")
