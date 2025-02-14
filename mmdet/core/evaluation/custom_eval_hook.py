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
        if eval_results is None:
            return None
        
        # Print per-class AP if available
        if 'classwise' in eval_results:
            try:
                headers = ['Category', 'AP', 'AP50', 'AP75']
                table_data = [headers]
                
                # Organize per-class results
                for item in eval_results['classwise']:
                    if len(item) >= 4:  # Making sure we have all metrics
                        category, ap, ap50, ap75 = item[:4]
                        table_data.append([
                            category,
                            f'{float(ap):0.3f}',
                            f'{float(ap50):0.3f}',
                            f'{float(ap75):0.3f}'
                        ])
                
                table = AsciiTable(table_data)
                print_log('\nPer-class evaluation:', logger=runner.logger)
                print_log('\n' + table.table, logger=runner.logger)
            except Exception as e:
                print_log(f'\nError in printing per-class results: {str(e)}', 
                         logger=runner.logger)

        # Print detailed metrics
        print_log('\nOverall metrics:', logger=runner.logger)
        for metric, value in eval_results.items():
            if isinstance(value, float):
                print_log(f'{metric}: {value:.4f}', logger=runner.logger)
            elif isinstance(value, str) and metric.endswith('copypaste'):
                print_log(f'{metric}: {value}', logger=runner.logger)

        # Print current losses
        if hasattr(runner, 'outputs') and 'log_vars' in runner.outputs:
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
            if eval_results is None:
                return None
            
            # Print per-class metrics
            if 'classwise' in eval_results:
                try:
                    headers = ['Category', 'AP', 'AP50', 'AP75']
                    table_data = [headers]
                    
                    # Organize per-class results
                    for item in eval_results['classwise']:
                        if len(item) >= 4:  # Making sure we have all metrics
                            category, ap, ap50, ap75 = item[:4]
                            table_data.append([
                                category,
                                f'{float(ap):0.3f}',
                                f'{float(ap50):0.3f}',
                                f'{float(ap75):0.3f}'
                            ])
                    
                    table = AsciiTable(table_data)
                    print_log('\nPer-class evaluation:', logger=runner.logger)
                    print_log('\n' + table.table, logger=runner.logger)
                except Exception as e:
                    print_log(f'\nError in printing per-class results: {str(e)}', 
                             logger=runner.logger)

            # Print detailed metrics
            print_log('\nOverall metrics:', logger=runner.logger)
            for metric, value in eval_results.items():
                if isinstance(value, float):
                    print_log(f'{metric}: {value:.4f}', logger=runner.logger)
                elif isinstance(value, str) and metric.endswith('copypaste'):
                    print_log(f'{metric}: {value}', logger=runner.logger)

            # Print current losses
            if hasattr(runner, 'outputs') and 'log_vars' in runner.outputs:
                print_log('\nCurrent Training Losses:', logger=runner.logger)
                for name, value in runner.outputs['log_vars'].items():
                    print_log(f'{name}: {value:.4f}', logger=runner.logger)

            return eval_results