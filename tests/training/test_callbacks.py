# tests/training/test_callbacks.py

import os
import sys
import tempfile
import torch
from pathlib import Path
from unittest.mock import Mock, MagicMock

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.training.callbacks import (
    EarlyStopping, ModelCheckpoint, LearningRateScheduler, 
    CallbackList, GradientMonitor, NaNDetector
)


class MockTrainer:
    """Mock trainer for testing callbacks."""
    
    def __init__(self):
        self.should_stop = False
        self.global_step = 0
        self.experiment = Mock()
        self.config = {
            'training': {
                'validation_freq': 6000,
                'save_freq': 6000,
                'patience': 30000
            }
        }
        
        # Create temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        self.dirs = {
            'checkpoint_dir': self.temp_dir
        }
        
        # Mock optimizer and model for gradient testing
        self.optimizer = Mock()
        self.model = Mock()
        self.model.named_parameters.return_value = [
            ('layer1.weight', torch.tensor([1.0, 2.0], requires_grad=True)),
            ('layer2.bias', torch.tensor([0.5], requires_grad=True))
        ]
        
        # Mock save_checkpoint method
        self.save_checkpoint = Mock()
    
    def cleanup(self):
        """Clean up temporary directory."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)


def test_early_stopping():
    """Test EarlyStopping callback with step-based logic."""
    print("Testing EarlyStopping...")
    
    trainer = MockTrainer()
    
    # Test with default patience from our config
    early_stop = EarlyStopping(
        monitor='val_total_loss',
        patience=30000,  # 30k steps as in base.yaml
        mode='min',
        verbose=False
    )
    
    # Initialize
    early_stop.on_train_begin(trainer)
    assert early_stop.wait_steps == 0
    assert early_stop.best is not None  # Should be inf for 'min' mode
    assert early_stop.last_validation_step == 0
    
    # Test improvement case - should reset wait
    logs_good = {'val_total_loss': 0.5}
    early_stop.on_step_end(trainer, 6000, logs_good)
    assert early_stop.wait_steps == 0
    assert early_stop.best == 0.5
    assert early_stop.last_validation_step == 6000
    assert not trainer.should_stop
    
    # Test no improvement case - should increment wait by step difference
    logs_bad = {'val_total_loss': 0.8}
    early_stop.on_step_end(trainer, 12000, logs_bad)
    # wait_steps += (12000 - 6000) = 6000
    assert early_stop.wait_steps == 6000
    assert early_stop.last_validation_step == 12000
    assert not trainer.should_stop
    
    # Continue with bad validations until patience exceeded
    # Need to accumulate 30000 steps total
    for step in [18000, 24000, 30000, 36000, 42000]:  # Each adds 6000 to wait_steps
        early_stop.on_step_end(trainer, step, logs_bad)
        if early_stop.wait_steps >= 30000:
            break
    
    assert trainer.should_stop  # Should stop after patience exceeded
    assert early_stop.wait_steps >= 30000
    
    # Test that callback ignores steps without validation metrics
    logs_no_val = {'train_loss': 0.3}
    early_stop.on_step_end(trainer, 48000, logs_no_val)
    # Should not change anything
    
    trainer.cleanup()
    print("✓ EarlyStopping test passed")


def test_model_checkpoint():
    """Test ModelCheckpoint callback."""
    print("Testing ModelCheckpoint...")
    
    trainer = MockTrainer()
    
    # Use constructor that matches actual implementation (no filepath parameter)
    checkpoint = ModelCheckpoint(
        monitor='val_total_loss',
        mode='min',
        save_best_only=True,
        verbose=False
    )
    
    # Initialize
    checkpoint.on_train_begin(trainer)
    assert checkpoint.best is not None
    
    # Test first save (should save as it's the first validation)
    logs1 = {'val_total_loss': 0.8}
    checkpoint.on_step_end(trainer, 6000, logs1)
    assert checkpoint.best == 0.8
    trainer.save_checkpoint.assert_called()
    
    # Reset mock
    trainer.save_checkpoint.reset_mock()
    
    # Test improvement (should save)
    logs2 = {'val_total_loss': 0.5}
    checkpoint.on_step_end(trainer, 12000, logs2)
    assert checkpoint.best == 0.5
    trainer.save_checkpoint.assert_called()
    
    # Reset mock
    trainer.save_checkpoint.reset_mock()
    
    # Test no improvement (should not save)
    logs3 = {'val_total_loss': 0.7}
    checkpoint.on_step_end(trainer, 18000, logs3)
    assert checkpoint.best == 0.5  # Should remain the best
    trainer.save_checkpoint.assert_not_called()
    
    # Test that callback ignores steps without validation metrics
    logs_no_val = {'train_loss': 0.3}
    checkpoint.on_step_end(trainer, 24000, logs_no_val)
    trainer.save_checkpoint.assert_not_called()
    
    trainer.cleanup()
    print("✓ ModelCheckpoint test passed")


def test_learning_rate_scheduler():
    """Test LearningRateScheduler callback."""
    print("Testing LearningRateScheduler...")
    
    trainer = MockTrainer()
    
    # Mock scheduler - add the __code__ attribute Mock lacks
    mock_scheduler = Mock()
    mock_scheduler.step.__code__ = Mock()
    mock_scheduler.step.__code__.co_varnames = ['self']  # No 'metrics' parameter
    
    # Test with default step_on='step' (matches actual implementation)
    lr_scheduler = LearningRateScheduler(
        scheduler=mock_scheduler,
        step_on='step',  # Use default value
        verbose=False
    )
    
    # Test step scheduling (should work with current implementation)
    logs = {'train_loss': 0.5}
    lr_scheduler.on_step_end(trainer, 1000, logs)
    mock_scheduler.step.assert_called_once()
    
    # Test batch scheduling (should not trigger since step_on='step')
    mock_scheduler.reset_mock()
    lr_scheduler.on_batch_end(trainer, 1000, logs)
    mock_scheduler.step.assert_not_called()  # Should not be called for step_on='step'
    
    # Test batch mode
    lr_scheduler.step_on = 'batch'
    lr_scheduler.on_batch_end(trainer, 2000, logs)
    mock_scheduler.step.assert_called_once()
    
    trainer.cleanup()
    print("✓ LearningRateScheduler test passed")


def test_callback_list():
    """Test CallbackList forwards calls correctly."""
    print("Testing CallbackList...")
    
    trainer = MockTrainer()
    
    # Create mock callbacks
    callback1 = Mock()
    callback2 = Mock()
    
    callback_list = CallbackList([callback1, callback2])
    
    # Test train begin
    callback_list.on_train_begin(trainer)
    callback1.on_train_begin.assert_called_once_with(trainer)
    callback2.on_train_begin.assert_called_once_with(trainer)
    
    # Test step end
    logs = {'val_total_loss': 0.5}
    callback_list.on_step_end(trainer, 6000, logs)
    callback1.on_step_end.assert_called_once_with(trainer, 6000, logs)
    callback2.on_step_end.assert_called_once_with(trainer, 6000, logs)
    
    # Test batch end
    callback_list.on_batch_end(trainer, 100, logs)
    callback1.on_batch_end.assert_called_once_with(trainer, 100, logs)
    callback2.on_batch_end.assert_called_once_with(trainer, 100, logs)
    
    trainer.cleanup()
    print("✓ CallbackList test passed")


def test_gradient_monitor():
    """Test GradientMonitor callback."""
    print("Testing GradientMonitor...")
    
    trainer = MockTrainer()
    
    grad_monitor = GradientMonitor(
        monitor_freq=10,
        verbose=False
    )
    
    # Create parameters with gradients for testing
    param1 = torch.tensor([1.0, 2.0], requires_grad=True)
    param2 = torch.tensor([0.5], requires_grad=True)
    param1.grad = torch.tensor([0.1, 0.2])
    param2.grad = torch.tensor([0.05])
    
    # Mock model.parameters() to return our test parameters
    trainer.model.parameters = Mock(return_value=[param1, param2])
    
    # Test monitoring (should work without errors)
    logs = {'train_loss': 0.5}
    for batch in range(20):  # Test multiple batches
        grad_monitor.on_batch_end(trainer, batch, logs)
    
    # Test that step_end does nothing
    grad_monitor.on_step_end(trainer, 1000, logs)
    
    trainer.cleanup()
    print("✓ GradientMonitor test passed")


def test_nan_detector():
    """Test NaNDetector callback."""
    print("Testing NaNDetector...")
    
    trainer = MockTrainer()
    
    nan_detector = NaNDetector(
        check_freq=5,
        raise_on_nan=False
    )
    
    # Test normal case (no NaN)
    logs_normal = {'loss': 0.5}
    param1 = torch.tensor([1.0, 2.0], requires_grad=True)
    param1.grad = torch.tensor([0.1, 0.2])
    
    # Mock model.named_parameters() for NaN detection
    trainer.model.named_parameters = Mock(return_value=[('layer1.weight', param1)])
    
    nan_detector.on_batch_end(trainer, 5, logs_normal)  # Should trigger check
    
    # Test NaN in loss
    logs_nan = {'loss': float('nan')}
    nan_detector.on_batch_end(trainer, 10, logs_nan)  # Should log error but not raise
    
    # Test that step_end does nothing
    nan_detector.on_step_end(trainer, 1000, logs_normal)
    
    trainer.cleanup()
    print("✓ NaNDetector test passed")


def run_all_tests():
    """Run all callback tests."""
    print("Running Callback Tests...")
    print("=" * 50)
    
    tests = [
        ("EarlyStopping", test_early_stopping),
        ("ModelCheckpoint", test_model_checkpoint),
        ("LearningRateScheduler", test_learning_rate_scheduler),
        ("CallbackList", test_callback_list),
        ("GradientMonitor", test_gradient_monitor),
        ("NaNDetector", test_nan_detector)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"✗ {test_name} test failed: {e}")
    
    print("=" * 50)
    print(f"Callback Tests: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All callback tests passed!")
    else:
        print(f"❌ {total - passed} tests failed")


if __name__ == "__main__":
    run_all_tests()