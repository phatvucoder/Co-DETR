# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import torch.distributed as dist
from mmcv.runner import DistEvalHook as BaseDistEvalHook
from mmcv.runner import EvalHook as BaseEvalHook
from terminaltables import AsciiTable
from mmcv.utils import print_log
from torch.nn.modules.batchnorm import _BatchNorm
import numpy as np

class CustomEvalHook(BaseEvalHook):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.latest_results = None

    def _format_metrics_table(self, eval_results, runner):
        """Format evaluation results into a nice table."""
        try:
            # Headers for the table
            headers = ['Class', 'Loss_cls', 'Loss_bbox', 'AP', 'AP50', 'AP75', 'AR', 'F1']
            table_data = [headers]
            
            # Get per-class results
            if 'classwise' in eval_results:
                results_per_class = {}
                for item in eval_results['classwise']:
                    class_name = item[0]
                    ap = float(item[1])
                    ap50 = float(item[2])
                    ap75 = float(item[3])
                    ar = float(item[4])
                    # Calculate F1 score (harmonic mean of precision and recall)
                    f1 = 2 * (ap50 * ar) / (ap50 + ar) if (ap50 + ar) > 0 else 0
                    
                    # Get class-specific losses if available
                    loss_cls = runner.outputs['log_vars'].get(f'd0.loss_cls_{class_name}', 0.0)
                    loss_bbox = runner.outputs['log_vars'].get(f'd0.loss_bbox_{class_name}', 0.0)
                    
                    results_per_class[class_name] = [
                        class_name,
                        f'{loss_cls:.3f}',
                        f'{loss_bbox:.3f}',
                        f'{ap:.3f}',
                        f'{ap50:.3f}',
                        f'{ap75:.3f}',
                        f'{ar:.3f}',
                        f'{f1:.3f}'
                    ]
                
                # Add per-class rows
                table_data.extend([results_per_class[k] for k in sorted(results_per_class.keys())])
                
                # Add mean values row
                mean_row = [
                    'Mean',
                    f'{runner.outputs["log_vars"].get("loss_cls", 0.0):.3f}',
                    f'{runner.outputs["log_vars"].get("loss_bbox", 0.0):.3f}',
                    f'{eval_results.get("bbox_mAP", 0.0):.3f}',
                    f'{eval_results.get("bbox_mAP_50", 0.0):.3f}',
                    f'{eval_results.get("bbox_mAP_75", 0.0):.3f}',
                    f'{eval_results.get("AR@100", 0.0):.3f}',
                    '-'  # No mean F1
                ]
                table_data.append(mean_row)
                
                table = AsciiTable(table_data)
                return table.table
            return ""
        except Exception as e:
            return f"Error formatting table: {str(e)}"

    def _do_evaluate(self, runner):
        """Perform evaluation and print detailed information."""
        if not self._should_evaluate(runner):
            return

        from mmdet.apis import single_gpu_test
        results = single_gpu_test(runner.model, self.dataloader, show=False)
        self.latest_results = results
        runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
        
        # Get evaluation results
        eval_results = self.evaluate(runner, results)
        if eval_results is None:
            return None

        # Format and print metrics table
        table = self._format_metrics_table(eval_results, runner)
        print_log('\nDetection Performance:\n' + table, logger=runner.logger)
        return eval_results

class CustomDistEvalHook(BaseDistEvalHook):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.latest_results = None

    def _format_metrics_table(self, eval_results, runner):
        """Format evaluation results into a nice table."""
        try:
            # Headers for the table
            headers = ['Class', 'Loss_cls', 'Loss_bbox', 'AP', 'AP50', 'AP75', 'AR', 'F1']
            table_data = [headers]
            
            # Get per-class results
            if 'classwise' in eval_results:
                results_per_class = {}
                for item in eval_results['classwise']:
                    class_name = item[0]
                    ap = float(item[1])
                    ap50 = float(item[2])
                    ap75 = float(item[3])
                    ar = float(item[4])
                    # Calculate F1 score
                    f1 = 2 * (ap50 * ar) / (ap50 + ar) if (ap50 + ar) > 0 else 0
                    
                    # Get class-specific losses if available
                    loss_cls = runner.outputs['log_vars'].get(f'd0.loss_cls_{class_name}', 0.0)
                    loss_bbox = runner.outputs['log_vars'].get(f'd0.loss_bbox_{class_name}', 0.0)
                    
                    results_per_class[class_name] = [
                        class_name,
                        f'{loss_cls:.3f}',
                        f'{loss_bbox:.3f}',
                        f'{ap:.3f}',
                        f'{ap50:.3f}',
                        f'{ap75:.3f}',
                        f'{ar:.3f}',
                        f'{f1:.3f}'
                    ]
                
                # Add per-class rows
                table_data.extend([results_per_class[k] for k in sorted(results_per_class.keys())])
                
                # Add mean values row
                mean_row = [
                    'Mean',
                    f'{runner.outputs["log_vars"].get("loss_cls", 0.0):.3f}',
                    f'{runner.outputs["log_vars"].get("loss_bbox", 0.0):.3f}',
                    f'{eval_results.get("bbox_mAP", 0.0):.3f}',
                    f'{eval_results.get("bbox_mAP_50", 0.0):.3f}',
                    f'{eval_results.get("bbox_mAP_75", 0.0):.3f}',
                    f'{eval_results.get("AR@100", 0.0):.3f}',
                    '-'  # No mean F1
                ]
                table_data.append(mean_row)
                
                table = AsciiTable(table_data)
                return table.table
            return ""
        except Exception as e:
            return f"Error formatting table: {str(e)}"

    def _do_evaluate(self, runner):
        """Perform evaluation with detailed metrics in distributed setting."""
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
            eval_results = self.evaluate(runner, results)
            if eval_results is None:
                return None
            
            # Format and print metrics table
            table = self._format_metrics_table(eval_results, runner)
            print_log('\nDetection Performance:\n' + table, logger=runner.logger)
            return eval_results