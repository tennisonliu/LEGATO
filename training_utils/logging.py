import json

def save_results(pretrain_loss, ft_loss, test_loss, auroc, acc, fname):
    res = {}
    res['pretrain'] = pretrain_loss
    res['ft'] = ft_loss
    res['test'] = test_loss
    res['auroc'] = auroc
    res['acc'] = acc
    res = json.dumps(res)
    f = open(f"{fname}.json", "w")
    f.write(res)
    f.close()
