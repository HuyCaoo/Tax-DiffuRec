"""
trainer.py — Combined: Category Embedding + Taxonomy Loss
==========================================================
Thay đổi so với trainer gốc:
  1. Dùng loss_combined() = L_CE + warmup_weight * L_taxonomy
  2. Warm-up 3 giai đoạn: warmup → ramp-up → full
  3. Log thêm loss_ce, loss_tax, cat_acc mỗi epoch
  4. Lưu best_model.pth ngay khi cải thiện
"""

import torch.nn as nn
import torch.optim as optim
import datetime
import torch
import numpy as np
import copy
import os
import pickle


def optimizers(model, args):
    if args.optimizer.lower() == 'adam':
        return optim.Adam(model.parameters(), lr=args.lr,
                          weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'sgd':
        return optim.SGD(model.parameters(), lr=args.lr,
                         weight_decay=args.weight_decay,
                         momentum=args.momentum)
    else:
        raise ValueError


def cal_hr(label, predict, ks):
    max_ks = max(ks)
    _, topk_predict = torch.topk(predict, k=max_ks, dim=-1)
    hit = label == topk_predict
    return [hit[:, :ks[i]].sum().item() / label.size()[0] for i in range(len(ks))]


def cal_ndcg(label, predict, ks):
    max_ks = max(ks)
    _, topk_predict = torch.topk(predict, k=max_ks, dim=-1)
    hit  = (label == topk_predict).int()
    ndcg = []
    for k in ks:
        max_dcg     = dcg(torch.tensor([1] + [0] * (k - 1)))
        predict_dcg = dcg(hit[:, :k])
        ndcg.append((predict_dcg / max_dcg).mean().item())
    return ndcg


def dcg(hit):
    log2 = torch.log2(torch.arange(1, hit.size()[-1] + 1) + 1).unsqueeze(0)
    return (hit / log2).sum(dim=-1)


def hrs_and_ndcgs_k(scores, labels, ks):
    metrics = {}
    ndcg = cal_ndcg(labels.clone().detach().to('cpu'),
                    scores.clone().detach().to('cpu'), ks)
    hr   = cal_hr(labels.clone().detach().to('cpu'),
                  scores.clone().detach().to('cpu'), ks)
    for k, ndcg_temp, hr_temp in zip(ks, ndcg, hr):
        metrics['HR@%d'   % k] = hr_temp
        metrics['NDCG@%d' % k] = ndcg_temp
    return metrics


def _get_state(model):
    return (model.module.state_dict()
            if isinstance(model, nn.DataParallel)
            else model.state_dict())


def _taxonomy_weight(epoch, warmup_epochs, rampup_epochs, base_alpha):
    """
    3 giai đoạn warm-up:
      [0, warmup)              → 0.0   (chỉ train CE)
      [warmup, warmup+rampup)  → tăng tuyến tính 0 → base_alpha
      [warmup+rampup, ...)     → base_alpha (ổn định)
    """
    if epoch < warmup_epochs:
        return 0.0
    elif epoch < warmup_epochs + rampup_epochs:
        progress = (epoch - warmup_epochs) / rampup_epochs
        return base_alpha * progress
    else:
        return base_alpha


def LSHT_inference(model_joint, args, data_loader):
    device = args.device
    model_joint = model_joint.to(device)
    with torch.no_grad():
        metrics_dict = {
            'HR@5': [], 'NDCG@5': [], 'HR@10': [],
            'NDCG@10': [], 'HR@20': [], 'NDCG@20': []
        }
        for test_batch in data_loader:
            test_batch = [x.to(device) for x in test_batch]
            _, rep_diffu, _, _, _, _ = model_joint(
                test_batch[0], test_batch[1], train_flag=False
            )
            scores  = model_joint.diffu_rep_pre(rep_diffu)
            metrics = hrs_and_ndcgs_k(scores, test_batch[1], [5, 10, 20])
            for k, v in metrics.items():
                metrics_dict[k].append(v)
    print({k: round(np.mean(v) * 100, 4) for k, v in metrics_dict.items()})


def model_train(tra_data_loader, val_data_loader, test_data_loader,
                model_joint, args, logger):

    epochs    = args.epochs
    device    = args.device
    metric_ks = args.metric_ks

    # Warm-up config
    warmup_epochs  = getattr(args, 'warmup_epochs',  20)
    rampup_epochs  = getattr(args, 'rampup_epochs',  20)
    base_alpha     = getattr(args, 'alpha_taxonomy', 0.1)

    model_joint = model_joint.to(device)
    # Đảm bảo taxonomy_loss_fn cũng trên đúng device
    if hasattr(model_joint, 'taxonomy_loss_fn'):
        model_joint.taxonomy_loss_fn = model_joint.taxonomy_loss_fn.to(device)

    if args.num_gpu > 1:
        model_joint = nn.DataParallel(model_joint)

    optimizer    = optimizers(model_joint, args)
    lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=args.decay_step, gamma=args.gamma
    )

    best_metrics_dict = {
        'Best_HR@5': 0, 'Best_NDCG@5': 0,
        'Best_HR@10': 0, 'Best_NDCG@10': 0,
        'Best_HR@20': 0, 'Best_NDCG@20': 0
    }
    best_epoch = {
        'Best_epoch_HR@5': 0,  'Best_epoch_NDCG@5': 0,
        'Best_epoch_HR@10': 0, 'Best_epoch_NDCG@10': 0,
        'Best_epoch_HR@20': 0, 'Best_epoch_NDCG@20': 0
    }
    bad_count  = 0
    best_model = copy.deepcopy(model_joint)
    history = {
        'epoch': [],
        'loss_total': [],
        'loss_ce': [],
        'loss_tax': [],
        'cat_acc': [],
        'tax_weight': [],
        'HR@5': [],
        'NDCG@5': [],
        'HR@10': [],
        'NDCG@10': [],
        'HR@20': [],
        'NDCG@20': [],
    }
    save_dir  = os.path.join(args.log_file, args.dataset)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(
        save_dir,
        f'best_model_{args.description}.pth'
    )

    history_path = os.path.join(
        save_dir,
        f'history_{args.description}.pkl'
    )

    result_path = os.path.join(
        save_dir,
        f'results_{args.description}.pkl'
    )

    for epoch_temp in range(epochs):
        print('Epoch: {}'.format(epoch_temp))
        logger.info('Epoch: {}'.format(epoch_temp))
        model_joint.train()

        # Tính taxonomy weight theo lịch warm-up
        tax_w = _taxonomy_weight(epoch_temp, warmup_epochs,
                                 rampup_epochs, base_alpha)

        flag_update  = 0
        ep_loss_tot  = 0.0
        ep_loss_ce   = 0.0
        ep_loss_tax  = 0.0
        ep_cat_acc   = 0.0
        n_batches    = 0

        for index_temp, train_batch in enumerate(tra_data_loader):
            train_batch = [x.to(device) for x in train_batch]
            optimizer.zero_grad()

            _, diffu_rep, _, _, _, _ = model_joint(
                train_batch[0], train_batch[1], train_flag=True
            )

            # Combined loss = L_CE + warm_weight * L_taxonomy
            loss_total, loss_ce, loss_tax = model_joint.loss_combined(
                diffu_rep, train_batch[1], warmup_weight=tax_w
            )

            loss_total.backward()
            optimizer.step()

            ep_loss_tot += loss_total.item()
            ep_loss_ce  += loss_ce.item()
            ep_loss_tax += loss_tax.item()
            ep_cat_acc  += model_joint.category_accuracy(
                diffu_rep.detach(), train_batch[1]
            )
            n_batches   += 1

            if index_temp % int(len(tra_data_loader) / 5 + 1) == 0:
                msg = (
                    '[%d/%d] Loss: %.4f | CE: %.4f | Tax: %.4f | tax_w: %.3f'
                    % (index_temp, len(tra_data_loader),
                       loss_total.item(), loss_ce.item(),
                       loss_tax.item(), tax_w)
                )
                print(msg)
                logger.info(msg)

        # Epoch summary
        summary = (
            'Epoch %d | Loss: %.4f | CE: %.4f | Tax: %.4f | '
            'CatAcc: %.4f | tax_w: %.3f'
            % (epoch_temp,
               ep_loss_tot / n_batches,
               ep_loss_ce  / n_batches,
               ep_loss_tax / n_batches,
               ep_cat_acc  / n_batches,
               tax_w)
        )
        print(summary)
        logger.info(summary)
        history['epoch'].append(epoch_temp)

        history['loss_total'].append(ep_loss_tot / n_batches)
        history['loss_ce'].append(ep_loss_ce / n_batches)
        history['loss_tax'].append(ep_loss_tax / n_batches)

        history['cat_acc'].append(ep_cat_acc / n_batches)

        history['tax_weight'].append(tax_w)
        lr_scheduler.step()

        # ── Validation ──────────────────────────────────────────────────
        if epoch_temp != 0 and epoch_temp % args.eval_interval == 0:
            print('start predicting: ', datetime.datetime.now())
            logger.info('start predicting: {}'.format(datetime.datetime.now()))
            model_joint.eval()

            with torch.no_grad():
                metrics_dict = {
                    'HR@5': [], 'NDCG@5': [], 'HR@10': [],
                    'NDCG@10': [], 'HR@20': [], 'NDCG@20': []
                }
                for val_batch in val_data_loader:
                    val_batch = [x.to(device) for x in val_batch]
                    _, rep_diffu, _, _, _, _ = model_joint(
                        val_batch[0], val_batch[1], train_flag=False
                    )
                    scores  = model_joint.diffu_rep_pre(rep_diffu)
                    metrics = hrs_and_ndcgs_k(scores, val_batch[1], metric_ks)
                    for k, v in metrics.items():
                        metrics_dict[k].append(v)
            metric_means = {}
            for key_temp, values_temp in metrics_dict.items():
                
                values_mean = round(np.mean(values_temp) * 100, 4)
                metric_means[key_temp] = values_mean
                if values_mean > best_metrics_dict['Best_' + key_temp]:
                    flag_update = 1
                    bad_count   = 0
                    best_metrics_dict['Best_' + key_temp]    = values_mean
                    best_epoch['Best_epoch_' + key_temp] = epoch_temp
            for k, v in metric_means.items():
                history[k].append(v)
            if flag_update == 0:
                bad_count += 1
            else:
                print(best_metrics_dict)
                print(best_epoch)
                logger.info(best_metrics_dict)
                logger.info(best_epoch)
                best_model = copy.deepcopy(model_joint)
                torch.save(_get_state(best_model), save_path)
                print(f"✅ Saved → {save_path}  (epoch {epoch_temp})")
                logger.info(f"Saved → {save_path}  (epoch {epoch_temp})")

            if bad_count >= args.patience:
                print(f"Early stopping tại epoch {epoch_temp}")
                logger.info(f"Early stopping tại epoch {epoch_temp}")
                break

    logger.info(best_metrics_dict)
    logger.info(best_epoch)

    if args.eval_interval > epochs:
        best_model = copy.deepcopy(model_joint)
        torch.save(_get_state(best_model), save_path)
        print(f"✅ Final model saved → {save_path}")

    # ── Test ────────────────────────────────────────────────────────────
    top_100_item = []
    with torch.no_grad():
        test_metrics_dict      = {
            'HR@5': [], 'NDCG@5': [], 'HR@10': [],
            'NDCG@10': [], 'HR@20': [], 'NDCG@20': []
        }
        test_metrics_dict_mean = {}
        for test_batch in test_data_loader:
            test_batch = [x.to(device) for x in test_batch]
            _, rep_diffu, _, _, _, _ = best_model(
                test_batch[0], test_batch[1], train_flag=False
            )
            scores = best_model.diffu_rep_pre(rep_diffu)

            _, indices = torch.topk(scores, k=100)
            top_100_item.append(indices)

            metrics = hrs_and_ndcgs_k(scores, test_batch[1], metric_ks)
            for k, v in metrics.items():
                test_metrics_dict[k].append(v)

    for key_temp, values_temp in test_metrics_dict.items():
        test_metrics_dict_mean[key_temp] = round(np.mean(values_temp) * 100, 4)

    print('Test------------------------------------------------------')
    logger.info('Test------------------------------------------------------')
    print(test_metrics_dict_mean)
    logger.info(test_metrics_dict_mean)
    print('Best Eval---------------------------------------------------------')
    logger.info('Best Eval---------------------------------------------------------')
    print(best_metrics_dict)
    print(best_epoch)
    logger.info(best_metrics_dict)
    logger.info(best_epoch)
    print(args)
    print(f"\n📦 Model đã lưu tại: {save_path}")

    if args.diversity_measure:
        path_data = '../datasets/data/category/' + args.dataset + '/id_category_dict.pkl'
        with open(path_data, 'rb') as f:
            id_category_dict = pickle.load(f)
        id_top_100 = torch.cat(top_100_item, dim=0).tolist()
        category_list_100 = []
        for id_top_100_temp in id_top_100:
            category_temp_list = []
            for id_temp in id_top_100_temp:
                category_temp_list.append(id_category_dict[id_temp])
            category_list_100.append(category_temp_list)
        path_data_category = (
            '../datasets/data/category/' + args.dataset
            + '/DiffuRec_top100_category.pkl'
        )
        with open(path_data_category, 'wb') as f:
            pickle.dump(category_list_100, f)
    # Save history
    with open(history_path, 'wb') as f:
        pickle.dump(history, f)

    print(f"📈 History saved → {history_path}")

    # Save final results
    final_results = {
        'best_metrics': best_metrics_dict,
        'best_epoch': best_epoch,
        'test_results': test_metrics_dict_mean,
        'args': vars(args)
    }

    with open(result_path, 'wb') as f:
        pickle.dump(final_results, f)

    print(f"📊 Results saved → {result_path}")
    return best_model, test_metrics_dict_mean
