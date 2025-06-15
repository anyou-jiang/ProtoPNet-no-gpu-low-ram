import time
import torch

from helpers import list_of_distances, make_one_hot

def _train_or_test(model, dataloader, optimizer=None, class_specific=True, use_l1_mask=True,
                   coefs=None, log=print):
    '''
    model: the multi-gpu model
    dataloader:
    optimizer: if None, will be test evaluation
    '''
    is_train = optimizer is not None
    start = time.time()
    n_examples = 0
    n_correct = 0
    n_batches = 0
    total_cross_entropy = 0
    total_cluster_cost = 0
    # separation cost is meaningful only for class_specific
    total_separation_cost = 0
    total_avg_separation_cost = 0

    for i, (image, label) in enumerate(dataloader):
        # input = image.cuda()
        # target = label.cuda()
        input = image # .cuda()
        target = label # .cuda()
        log('\t i={0}'.format(i))
        # torch.enable_grad() has no effect outside of no_grad()
        grad_req = torch.enable_grad() if is_train else torch.no_grad()
        log('\t with grad_req:')
        with grad_req:
            # nn.Module has implemented __call__() function
            # so no need to call .forward
            log('\t output, min_distances = model(input)')
            output, min_distances = model(input)

            # compute loss
            log('\t cross_entropy = torch.nn.functional.cross_entropy(output, target)')
            cross_entropy = torch.nn.functional.cross_entropy(output, target)

            if class_specific:
                log('\t max_dist = (model.module.prototype_shape[1]')
                max_dist = (model.module.prototype_shape[1]
                            * model.module.prototype_shape[2]
                            * model.module.prototype_shape[3])

                # prototypes_of_correct_class is a tensor of shape batch_size * num_prototypes
                # calculate cluster cost
                # prototypes_of_correct_class = torch.t(model.module.prototype_class_identity[:,label]).cuda()
                log('\t prototypes_of_correct_class = torch.t(model.module.prototype_class_identity[:,label])')
                prototypes_of_correct_class = torch.t(model.module.prototype_class_identity[:,label])
                log('\t inverted_distances, _ = torch.max((max_dist - min_distances) * prototypes_of_correct_class, dim=1)')
                inverted_distances, _ = torch.max((max_dist - min_distances) * prototypes_of_correct_class, dim=1)
                log('\t cluster_cost = torch.mean(max_dist - inverted_distances)')
                cluster_cost = torch.mean(max_dist - inverted_distances)

                # calculate separation cost
                log('\t prototypes_of_wrong_class = 1 - prototypes_of_correct_class')
                prototypes_of_wrong_class = 1 - prototypes_of_correct_class
                log('\t inverted_distances_to_nontarget_prototypes, _ = ')
                inverted_distances_to_nontarget_prototypes, _ = \
                    torch.max((max_dist - min_distances) * prototypes_of_wrong_class, dim=1)
                log('\t separation_cost = torch.mean(max_dist - inverted_distances_to_nontarget_prototypes)')
                separation_cost = torch.mean(max_dist - inverted_distances_to_nontarget_prototypes)

                # calculate avg cluster cost
                log('\t avg_separation_cost = ')
                avg_separation_cost = \
                    torch.sum(min_distances * prototypes_of_wrong_class, dim=1) / torch.sum(prototypes_of_wrong_class, dim=1)
                log('\t avg_separation_cost = torch.mean(avg_separation_cost)')
                avg_separation_cost = torch.mean(avg_separation_cost)
                
                if use_l1_mask:
                    #l1_mask = 1 - torch.t(model.module.prototype_class_identity).cuda()
                    log('\t l1_mask = 1 - torch.t(model.module.prototype_class_identity) ')
                    l1_mask = 1 - torch.t(model.module.prototype_class_identity) 
                    log('\t l1 = (model.module.last_layer.weight * l1_mask).norm(p=1)')
                    l1 = (model.module.last_layer.weight * l1_mask).norm(p=1)
                else:
                    log('\t l1 = model.module.last_layer.weight.norm(p=1) ')
                    l1 = model.module.last_layer.weight.norm(p=1) 

            else:
                log('\t min_distance, _ = torch.min(min_distances, dim=1)')
                min_distance, _ = torch.min(min_distances, dim=1)
                log('\t luster_cost = torch.mean(min_distance)')
                cluster_cost = torch.mean(min_distance)
                log('\t l1 = model.module.last_layer.weight.norm(p=1)')
                l1 = model.module.last_layer.weight.norm(p=1)

            # evaluation statistics
            log('\t _, predicted = torch.max(output.data, 1)')
            _, predicted = torch.max(output.data, 1)
            log('\t n_examples += target.size(0)')
            n_examples += target.size(0)
            log('\t n_correct += (predicted == target).sum().item()')
            n_correct += (predicted == target).sum().item()

            n_batches += 1
            total_cross_entropy += cross_entropy.item()
            total_cluster_cost += cluster_cost.item()
            total_separation_cost += separation_cost.item()
            total_avg_separation_cost += avg_separation_cost.item()

        # compute gradient and do SGD step
        if is_train:
            if class_specific:
                if coefs is not None:
                    loss = (coefs['crs_ent'] * cross_entropy
                          + coefs['clst'] * cluster_cost
                          + coefs['sep'] * separation_cost
                          + coefs['l1'] * l1)
                else:
                    loss = cross_entropy + 0.8 * cluster_cost - 0.08 * separation_cost + 1e-4 * l1
            else:
                if coefs is not None:
                    loss = (coefs['crs_ent'] * cross_entropy
                          + coefs['clst'] * cluster_cost
                          + coefs['l1'] * l1)
                else:
                    loss = cross_entropy + 0.8 * cluster_cost + 1e-4 * l1
            log('\t optimizer.zero_grad()')
            optimizer.zero_grad()
            log('\t loss.backward()')
            loss.backward()
            log('\t optimizer.step()')
            optimizer.step()

        del input
        del target
        del output
        del predicted
        del min_distances

    end = time.time()

    log('\ttime: \t{0}'.format(end -  start))
    log('\tcross ent: \t{0}'.format(total_cross_entropy / n_batches))
    log('\tcluster: \t{0}'.format(total_cluster_cost / n_batches))
    if class_specific:
        log('\tseparation:\t{0}'.format(total_separation_cost / n_batches))
        log('\tavg separation:\t{0}'.format(total_avg_separation_cost / n_batches))
    log('\taccu: \t\t{0}%'.format(n_correct / n_examples * 100))
    log('\tl1: \t\t{0}'.format(model.module.last_layer.weight.norm(p=1).item()))
    p = model.module.prototype_vectors.view(model.module.num_prototypes, -1)# .cpu()
    log('\tp')
    # with torch.no_grad():
    #     p_avg_pair_dist = torch.mean(list_of_distances(p, p))
    # log('\tp dist pair: \t{0}'.format(p_avg_pair_dist.item()))

    return n_correct / n_examples


def train(model, dataloader, optimizer, class_specific=False, coefs=None, log=print):
    assert(optimizer is not None)
    
    log('\ttrain')
    model.train()
    log('\t after model train()')
    return _train_or_test(model=model, dataloader=dataloader, optimizer=optimizer,
                          class_specific=class_specific, coefs=coefs, log=log)


def test(model, dataloader, class_specific=False, log=print):
    log('\ttest')
    model.eval()
    return _train_or_test(model=model, dataloader=dataloader, optimizer=None,
                          class_specific=class_specific, log=log)


def last_only(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = False
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = False
    model.module.prototype_vectors.requires_grad = False
    for p in model.module.last_layer.parameters():
        p.requires_grad = True
    
    log('\tlast layer')


def warm_only(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = False
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = True
    model.module.prototype_vectors.requires_grad = True
    for p in model.module.last_layer.parameters():
        p.requires_grad = True
    
    log('\twarm')


def joint(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = True
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = True
    model.module.prototype_vectors.requires_grad = True
    for p in model.module.last_layer.parameters():
        p.requires_grad = True
    
    log('\tjoint')
