"""
Various utility functions.
The code may be quite messy.
"""

import os
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from evaluators import IoUEvaluator as Evaluator


COLORS1 = [(148, 255, 181),
           (66, 102, 0), (116, 10, 155), (94, 241, 242), (0, 153, 143),
           (0, 255, 211), (128, 128, 128), (143, 124, 0), (157, 204, 0),
           (194, 0, 136), (255, 164, 5), (255, 168, 187), (255, 0, 16),
           (0, 153, 143), (224, 255, 102), (153, 0, 0), (255, 255, 128),
           (255, 255, 0), (255, 80, 5)]

VOC_COLORMAP = [[224, 224, 224], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128]]


def plot_learning_curve(step_losses, step_scores, eval_losses=None, eval_scores=None,
                        name='Score', loss_threshold=10, mode='all', img_dir='.', exp_idx=-1,
                        annotations=None, start_step=0, validation_frequency=None):
    plt.style.use('seaborn')
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    if eval_losses is None:
        axes[0].plot(np.arange(start_step + 1, len(step_losses) + start_step + 1), step_losses,
                     color='y', marker='', label='Training')
        axes[0].set_ylabel('Training Loss')
        # axes[0].set_xlabel('Number of Iterations')
        axes[0].set_ylim(0.0, min(loss_threshold, max(step_losses)))
    else:
        if validation_frequency is None:
            axes[0].plot(np.arange(1, len(eval_losses) + 1), eval_losses,
                         color='r', marker='', label='Validation')
            axes[0].set_ylabel('Validation Loss')
            # axes[0].set_xlabel('Number of Evaluations')
        else:
            axes[0].plot(np.arange(start_step + 1, len(step_losses) + start_step + 1), step_losses,
                         color='y', marker='', label='Training')
            axes[0].plot(np.arange(start_step + validation_frequency,
                                   start_step + (len(eval_losses) + 1)*validation_frequency,
                                   validation_frequency),
                         eval_losses, color='r', marker='', label='Validation')
            axes[0].set_ylabel('Loss')
            # if validation_frequency == 1:
            #     axes[0].set_xlabel('Number of Epochs')
            # else:
            #     axes[0].set_xlabel('Number of Iterations')
        axes[0].set_ylim(0.0, min(loss_threshold, max(max(step_losses), max(eval_losses))))
    axes[0].legend(loc='lower left')
    axes[0].grid(True)

    if eval_scores is None:
        final_score = step_scores[-1]
        if mode.lower == 'min':
            best_score = min(step_scores)
        else:
            best_score = max(step_scores)
        axes[1].set_title('Final Training {}: {:.4f} (Best: {:.4f})'.format(name, final_score, best_score))

        axes[1].plot(np.arange(start_step + 1, len(step_scores) + start_step + 1), step_scores,
                     color='y', marker='', label='Training')
        axes[1].set_ylabel('Training ' + name)
        axes[1].set_xlabel('Number of Iterations')
        if annotations is not None:
            if len(annotations) > 0:
                x, y = zip(*annotations)
                axes[1].plot(x, y, color='b', marker='o', linestyle='', label='Checkpoints')
                for i, (x, y) in enumerate(annotations):
                    axes[1].annotate('{:.4f}'.format(y), xy=(x, y), xytext=(0, -4 + (-1)**i*9), ha='center',
                                     textcoords='offset points', color='b', weight='bold')
    else:
        final_score = eval_scores[-1]
        if mode.lower == 'min':
            best_score = min(eval_scores)
        else:
            best_score = max(eval_scores)
        axes[1].set_title('Final Validation {}: {:.4f} (Best: {:.4f})'.format(name, final_score, best_score))

        if validation_frequency is None:
            axes[1].plot(np.arange(1, len(eval_scores) + 1), eval_scores, color='r', marker='', label='Validation')
            axes[1].set_ylabel('Validation ' + name)
            axes[1].set_xlabel('Number of Evaluations')
        else:
            axes[1].plot(np.arange(start_step + 1, len(step_scores) + start_step + 1), step_scores,
                         color='y', marker='', label='Training')
            axes[1].plot(np.arange(start_step + validation_frequency,
                                   start_step + (len(eval_losses) + 1)*validation_frequency,
                                   validation_frequency),
                         eval_scores, color='r', marker='', label='Validation')
            axes[1].set_ylabel(name)
            if validation_frequency == 1:
                axes[1].set_xlabel('Number of Epochs')
            else:
                axes[1].set_xlabel('Number of Iterations')
            if annotations is not None:
                if len(annotations) > 0:
                    x, y = zip(*annotations)
                    axes[1].plot(x, y, color='b', marker='o', linestyle='', label='Checkpoints')
                    for i, (x, y) in enumerate(annotations):
                        axes[1].annotate('{:.4f}'.format(y), xy=(x, y), xytext=(0, -4 + (-1)**i*9), ha='center',
                                         textcoords='offset points', color='b', weight='bold')
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend(loc='upper left')
    axes[1].grid(True)

    # Save plot as image file
    plot_img_filename = 'learning_curve-result{}.svg'.format(exp_idx)
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    fig.savefig(os.path.join(img_dir, plot_img_filename))

    # Save details as pkl file
    pkl_filename = 'learning_curve-result{}.pkl'.format(exp_idx)
    with open(os.path.join(img_dir, pkl_filename), 'wb') as fo:
        pkl.dump([step_losses, step_scores, eval_scores], fo)


def plot_class_results(images, y_true, y_pred=None, fault=None, num_rows=3, num_cols=3, shuffle=True,
                       class_names=None, save_dir=None):
    if y_pred is None:
        y_pred = y_true

    if y_true.shape[1] > 1:
        y_true = np.argmax(y_true, axis=1)

    if y_pred.shape[1] > 1:
        y_label = np.argmax(y_pred, axis=1)
        y_prob = y_pred
    else:
        y_label = np.around(y_pred).astype(int)
        y_prob = np.concatenate((1 - y_pred, y_pred), axis=1)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(9, 9))
    fig.subplots_adjust(left=0.075, right=0.95, bottom=0.05, top=0.9, wspace=0.3, hspace=0.3)

    num_images = images.shape[0]
    if fault is None:
        fig.suptitle('Results', fontsize=11)
        idx = list(range(num_images))
    else:
        if fault:
            idx = [i for i in range(num_images) if y_true[i] != y_label[i]]
            fig.suptitle('Incorrect Answers', fontsize=11)
        else:
            idx = [i for i in range(num_images) if y_true[i] == y_label[i]]
            fig.suptitle('Correct Answers', fontsize=11)

    if shuffle:
        idx = np.random.permutation(idx)

    i = 0
    for row in range(num_cols):
        for col in range(num_rows):
            if i >= len(idx):
                break
            n = idx[i]
            axes[row, col].imshow(images[n])
            if class_names is None:
                axes[row, col].set_title('{:6d}. A: class {}, {:.2%}\n(class {}, {:.2%}).'
                                         .format(n, y_label[n], y_prob[n, y_label[n]],
                                                 y_true[n], y_prob[n, y_true[n]]),
                                         fontsize=9)
            else:
                axes[row, col].set_title('{:6d}. A: {}, {:.2%}\n({}, {:.2%}).'
                                         .format(n, class_names[y_label[n]], y_prob[n, y_label[n]],
                                                 class_names[y_true[n]], y_prob[n, y_true[n]]),
                                         fontsize=9)
            axes[row, col].tick_params(axis='both', labelsize=6)
            i += 1

    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        print('')
        for i in range(num_images):
            fig_name = os.path.join(save_dir, '{}.jpg'.format(i))
            if not os.path.isfile(os.path.join(save_dir, '{}.jpg'.format(i))):
                fig = plt.figure()
                plt.imshow(images[i])
                if class_names is None:
                    fig.suptitle('{:6d}. A: class {}, {:.2%}\n(class {}, {:.2%}).'
                                 .format(i, y_label[i], y_prob[i, y_label[i]], y_true[i], y_prob[i, y_true[i]]))
                else:
                    fig.suptitle('{:6d}. A: {}, {:.2%}\n({}, {:.2%}).'
                                 .format(i, class_names[y_label[i]], y_prob[i, y_label[i]],
                                         class_names[y_true[i]], y_prob[i, y_true[i]]))
                fig.savefig(fig_name)
                plt.close(fig)

                if i % 200 == 0:
                    print('Saving result images... {:5}/{}'.format(i, num_images))

    # plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names=None, normalize=False, top_confused_classes=20,
                          figure_title='Confusion Matrix', cmap=plt.cm.Blues, max_classes=25):
    num_examples = np.sum(np.isclose(y_true.sum(axis=-1), 1))
    num_classes = y_true.shape[-1]
    if len(y_true.shape) > 1:
        y_true = np.argmax(y_true, axis=-1)
    if len(y_pred.shape) > 1:
        y_pred = np.argmax(y_pred, axis=-1)

    correct_examples = np.sum(y_true == y_pred).astype(np.float32)
    accuracy = correct_examples/num_examples

    cm = confusion_matrix(y_true, y_pred)

    if top_confused_classes is not None:
        confused_classes = []
        for i in range(num_classes):
            total = cm[i].sum()
            for j in range(num_classes):
                if i == j:
                    wrong = 0.0
                else:
                    wrong = float(cm[i, j])/total
                confused_classes.append((wrong, (i, j)))
                confused_classes = sorted(confused_classes, key=lambda cls: cls[0], reverse=True)
                confused_classes = confused_classes[:top_confused_classes]

        values = []
        pairs = []
        for confused in confused_classes:
            values.append(confused[0])
            pairs.append('{}\n->{}'.format(class_names[confused[1][0]], class_names[confused[1][1]]))
        y_max = 1.1*max(values)

        fig, axes = plt.subplots(figsize=(14, 7))
        fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.35)
        idx = np.arange(len(values))
        axes.bar(idx, values, align='center', alpha=0.5)
        axes.grid(alpha=0.25)
        axes.set(xticks=idx,
                 xticklabels=pairs,
                 ylim=[0, y_max],
                 title='Top-{} Confused Classes'.format(top_confused_classes))
        plt.setp(axes.get_xticklabels(), rotation=60, ha='right', rotation_mode='anchor', va='top', size='small')
        for i, v in enumerate(values):
            axes.text(i + 0.05, v + 0.02*y_max, '{:.2%}'.format(v), ha='center', size='small')

    if normalize:
        cm = cm.astype('float')/cm.sum(axis=1, keepdims=True)

    too_many_classes = len(class_names) > max_classes
    if class_names is None or too_many_classes:
        class_names = []
        for i in range(num_classes):
            class_names.append(str(i))

    fig, axes = plt.subplots(figsize=(10, 8))
    img = axes.imshow(cm, interpolation='nearest', cmap=cmap)
    axes.figure.colorbar(img, ax=axes)
    axes.set(xticks=np.arange(cm.shape[1]),
             yticks=np.arange(cm.shape[0]),
             xticklabels=class_names, yticklabels=class_names,
             title=figure_title + " (accuracy: {:.2%})".format(accuracy),
             ylabel='True Labels',
             xlabel='Predicted Labels')
    plt.setp(axes.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    axes.set_xticks(np.arange(cm.shape[1] - 1) + 0.5, minor=True)
    axes.set_yticks(np.arange(cm.shape[0] - 1) + 0.5, minor=True)
    axes.grid(which='minor', alpha=0.5)

    if too_many_classes:
        print('Too many classes to show for the confusion matrix.')
    else:
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                axes.text(j, i, format(cm[i, j], fmt),
                          ha="center", va="center",
                          color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()

    return cm


def plot_features(features, y_true, y_pred, num_rows=3, num_cols=3, class_names=None, figure_title='Features'):
    if y_true.shape[0] > 1:
        y_true = np.argmax(y_true, axis=0)
    if y_pred.shape[0] > 1:
        y_label = np.argmax(y_pred, axis=0)
        y_prob = y_pred
    else:
        y_label = np.around(y_pred).astype(int)
        y_prob = np.array([1 - y_pred, y_pred])

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(9, 9))
    fig.subplots_adjust(left=0.075, right=0.95, bottom=0.05, top=0.9, wspace=0.3, hspace=0.3)
    if class_names is None:
        fig.suptitle(figure_title + '. A: class {}, {:.2%} (class {}, {:.2%}).'.format(y_label, y_prob[y_label],
                                                                                       y_true, y_prob[y_true]))
    else:
        fig.suptitle(figure_title + '. A: {}, {:.2%} ({}, {:.2%}).'.format(class_names[y_label], y_prob[y_label],
                                                                           class_names[y_true], y_prob[y_true]))

    i = 0
    for row in range(num_cols):
        for col in range(num_rows):
            axes[row, col].imshow(features[:, :, i], cmap='gray')
            axes[row, col].tick_params(axis='both', labelsize=6)
            i += 1
            if i >= features.shape[2]:
                break

    # plt.show()


def show_features(features, group_features=True, num_rows=3, num_cols=3, figure_title='Figure'):
    if group_features:
        num_images = features.shape[-1]//3
        images = np.empty([num_images] + list(features.shape[0:2]) + [3])
        for i in range(num_images):
            for j in range(3):
                images[i, :, :, j] = features[:, :, 3*i + j]
            images[i] = images[i] - np.amin(images[i])
            images[i] = images[i]/np.amax(images[i])
    else:
        num_images = features.shape[-1]
        images = np.empty([num_images] + list(features.shape[0:2]))
        for i in range(num_images):
            images[i] = features[:, :, i]
            images[i] = images[i] - np.amin(images[i])
            images[i] = images[i]/np.amax(images[i])

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(9, 9))
    fig.subplots_adjust(left=0.075, right=0.95, bottom=0.05, top=0.9, wspace=0.3, hspace=0.3)
    fig.suptitle(figure_title)

    i = 0
    for row in range(num_cols):
        for col in range(num_rows):
            axes[row, col].imshow(images[i])
            axes[row, col].tick_params(axis='both', labelsize=6)
            i += 1
            if i >= images.shape[0]:
                break


def imshow(image, y_true=None, y_pred=None, class_names=None, figure_title='Figure'):
    fig = plt.figure()
    plt.imshow(image)
    if y_true is not None and y_pred is not None:
        if y_true.shape[0] > 1:
            y_true = np.argmax(y_true, axis=0)
        if y_pred.shape[0] > 1:
            y_label = np.argmax(y_pred, axis=0)
            y_prob = y_pred
        else:
            y_label = np.around(y_pred).astype(int)
            y_prob = np.array([1 - y_pred, y_pred])

        if class_names is None:
            fig.suptitle(figure_title + '. A: class {}, {:.2%} (class {}, {:.2%}).'.format(y_label, y_prob[y_label],
                                                                                           y_true, y_prob[y_true]))
        else:
            fig.suptitle(figure_title + '. A: {}, {:.2%} ({}, {:.2%}).'.format(class_names[y_label], y_prob[y_label],
                                                                               class_names[y_true], y_prob[y_true]))
    else:
        fig.suptitle(figure_title)


def imshow_subplot(images, num_rows=3, num_cols=3, figure_title='Figure'):
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(9, 9))
    fig.subplots_adjust(left=0.075, right=0.95, bottom=0.05, top=0.9, wspace=0.3, hspace=0.3)
    fig.suptitle(figure_title)

    i = 0
    for row in range(num_cols):
        for col in range(num_rows):
            axes[row, col].imshow(images[i])
            axes[row, col].tick_params(axis='both', labelsize=6)
            i += 1
            if i >= images.shape[0]:
                break


def plot_seg_results(images, y_true, y_pred=None, num_rows=3, num_cols=3, colors=None, save_dir=None):
    if y_pred is None:
        y_pred = y_true

    num_classes = y_true.shape[-1]
    num_images = images.shape[0]
    images = (images*255).astype(np.int)

    if colors is None:
        mask_true = (seg_labels_to_images(y_true)*255).astype(np.int)
    else:
        mask_true = np.zeros(images.shape, dtype=np.int)
        y_t = y_true.argmax(axis=-1)
        v_t = y_true.sum(axis=-1) == 0
        for i in range(1, num_classes):
            mask_true[np.where(y_t == i)] = colors[i]
        mask_true[np.where(v_t)] = colors[0]

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(9, 9))
    fig.subplots_adjust(left=0.075, right=0.95, bottom=0.05, top=0.9, wspace=0.3, hspace=0.3)
    fig.suptitle('Ground Truths')
    i = 0
    for row in range(num_cols):
        for col in range(num_rows):
            if i >= images.shape[0]:
                break
            axes[row, col].imshow(np.clip(images[i] + mask_true[i], 0, 255))
            axes[row, col].tick_params(axis='both', labelsize=6)
            i += 1

    evaluator = Evaluator()
    name = evaluator.name
    scores = []
    for y_t, y_p in zip(y_true, y_pred):
        y_t = np.expand_dims(y_t, axis=0)
        y_p = np.expand_dims(y_p, axis=0)
        scores.append(evaluator.score(y_t, y_p))

    if colors is None:
        mask_pred = (seg_labels_to_images(y_pred)*255).astype(np.int)
    else:
        mask_pred = np.zeros(images.shape, dtype=np.int)
        y_p = y_pred.argmax(axis=-1)
        for i in range(1, num_classes):
            mask_pred[np.where(y_p == i)] = colors[i]

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(9, 9))
    fig.subplots_adjust(left=0.075, right=0.95, bottom=0.05, top=0.9, wspace=0.3, hspace=0.3)
    fig.suptitle('Predictions')
    i = 0
    for row in range(num_cols):
        for col in range(num_rows):
            if i >= images.shape[0]:
                break
            axes[row, col].imshow(np.clip(images[i] + mask_pred[i], 0, 255))
            axes[row, col].tick_params(axis='both', labelsize=6)
            axes[row, col].set_title('{}:\n{:.4}'.format(name, scores[i]))
            axes[row, col].tick_params(axis='both', labelsize=6)
            i += 1

    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        print('')
        for i in range(num_images):
            fig_name = os.path.join(save_dir, '{}.jpg'.format(i))
            if not os.path.isfile(os.path.join(save_dir, '{}.jpg'.format(i))):
                fig = plt.figure()
                plt.imshow(np.clip(images[i] + mask_pred[i], 0, 255))
                fig.suptitle('{}:\n{:.4}'.format(name, scores[i]))
                fig.savefig(fig_name)
                plt.close(fig)

                if i % 200 == 0:
                    print('Saving result images... {:5}/{}'.format(i, num_images))


def seg_labels_to_images(y):
    num_classes = y.shape[-1]
    edge_color = 0.8
    code_r = [1, 0, 0, 1, 1, 0, .8, 1, .6, .6]
    code_g = [0, 1, 0, 1, 0, 1, .8, .6, 1, .6]
    code_b = [0, 0, 1, 0, 1, 1, .8, .6, .6, 1]
    code_length = len(code_r)
    color_base = (num_classes - 1)//code_length + 1
    color_coeff = edge_color/color_base

    class_inds = np.arange(num_classes, dtype=np.float32)
    reds = (class_inds + code_length - 1)//code_length*color_coeff
    greens = (class_inds + code_length - 1)//code_length*color_coeff
    blues = (class_inds + code_length - 1)//code_length*color_coeff
    for i in range(1, num_classes):
        idx = (i - 1) % code_length
        reds[i] = reds[i]*code_r[idx]
        greens[i] = greens[i]*code_g[idx]
        blues[i] = blues[i]*code_b[idx]

    ignore = 1.0 - np.isclose(np.sum(y, axis=-1, keepdims=True), 1.0)
    r = np.sum(y*reds, axis=-1, keepdims=True)
    g = np.sum(y*greens, axis=-1, keepdims=True)
    b = np.sum(y*blues, axis=-1, keepdims=True)
    y = np.concatenate([r, g, b], axis=-1) + ignore.astype(np.float32)*edge_color

    return y
