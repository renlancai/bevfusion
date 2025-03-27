import mmcv
import torch


def single_gpu_test(model, data_loader):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    i = 0
    for data in data_loader:
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
            i = i + 1
        results.extend(result)
        # if (i > 100):
        #     break
        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()
    return results
