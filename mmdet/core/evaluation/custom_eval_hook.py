# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import torch.distributed as dist
from mmcv.runner import DistEvalHook as BaseDistEvalHook
from mmcv.runner import EvalHook as BaseEvalHook
from terminaltables import AsciiTable
from mmcv.utils import print_log
from torch.nn.modules.batchnorm import _BatchNorm

class CustomEvalHook(BaseEvalHook):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.latest_results = None

    def _do_evaluate(self, runner):
        """Perform evaluation and print detailed information."""
        if not self._should_evaluate(runner):
            return

        from mmdet.apis import single_gpu_test
        results = single_gpu_test(runner.model, self.dataloader, show=False)
        self.latest_results = results
        runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
        
        # Get all evaluation metrics
        eval_results = self.evaluate(runner, results)
        
        # Print per-class AP if available
        if 'classwise' in eval_results:
            headers = ['Category', 'AP']
            table_data = [headers]
            for category, ap in eval_results['classwise']:
                table_data.append([category, f'{float(ap):0.3f}'])
            table = AsciiTable(table_data)
            print_log('\n' + table.table, logger=runner.logger)

        # Print detailed metrics
        for metric, value in eval_results.items():
            if isinstance(value, float):
                print_log(f'\n{metric}: {value:.4f}', logger=runner.logger)
            elif isinstance(value, str) and metric.endswith('copypaste'):
                print_log(f'\n{metric}: {value}', logger=runner.logger)

        # Print current losses
        print_log('\nCurrent Training Losses:', logger=runner.logger)
        for name, value in runner.outputs['log_vars'].items():
            print_log(f'{name}: {value:.4f}', logger=runner.logger)

        return eval_results

class CustomDistEvalHook(BaseDistEvalHook):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.latest_results = None

    def _do_evaluate(self, runner):
        """Perform evaluation with detailed metrics in distributed setting."""
        # Synchronize BatchNorm statistics
        if self.broadcast_bn_buffer:
            model = runner.model
            for name, module in model.named_modules():
                if isinstance(module, _BatchNorm) and module.track_running_stats:
                    dist.broadcast(module.running_var, 0)
                    dist.broadcast(module.running_mean, 0)

        if not self._should_evaluate(runner):
            return

        tmpdir = self.tmpdir
        if tmpdir is None:
            tmpdir = osp.join(runner.work_dir, '.eval_hook')

        from mmdet.apis import multi_gpu_test
        results = multi_gpu_test(
            runner.model,
            self.dataloader,
            tmpdir=tmpdir,
            gpu_collect=self.gpu_collect)
        
        self.latest_results = results
        
        if runner.rank == 0:
            print_log('\n', logger=runner.logger)
            
            # Get evaluation results
            eval_results = self.evaluate(runner, results)
            
            # Print per-class metrics
            if 'classwise' in eval_results:
                headers = ['Category', 'AP']
                table_data = [headers]
                for category, ap in eval_results['classwise']:
                    table_data.append([category, f'{float(ap):0.3f}'])
                table = AsciiTable(table_data)
                print_log('\n' + table.table, logger=runner.logger)

            # Print detailed metrics
            print_log('\nDetailed Metrics:', logger=runner.logger)
            for metric, value in eval_results.items():
                if isinstance(value, float):
                    print_log(f'{metric}: {value:.4f}', logger=runner.logger)
                elif isinstance(value, str) and metric.endswith('copypaste'):
                    print_log(f'{metric}: {value}', logger=runner.logger)

            # Print current losses
            print_log('\nCurrent Training Losses:', logger=runner.logger)
            for name, value in runner.outputs['log_vars'].items():
                print_log(f'{name}: {value:.4f}', logger=runner.logger)

            return eval_results