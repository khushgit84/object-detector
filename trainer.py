import os
import shutil
import threading
import time
from typing import Dict, List, Optional, Any
import yaml
from ultralytics import YOLO
import torch

class CustomTrainer:
    def __init__(self, datasets_dir: str = "datasets", models_dir: str = "models"):
        self.datasets_dir = os.path.abspath(datasets_dir)
        self.models_dir = os.path.abspath(models_dir)
        os.makedirs(self.datasets_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)

        self.lock = threading.Lock()
        self.is_training = False
        self.training_thread: Optional[threading.Thread] = None
        self.current_job: Dict[str, Any] = {
            "status": "idle",
            "progress": 0,
            "epoch": 0,
            "total_epochs": 0,
            "loss": 0.0,
            "map50": 0.0,
            "log": [],
            "error": None,
            "output_model": None
        }

    def get_status(self) -> Dict[str, Any]:
        """Returns the current training job status."""
        with self.lock:
            return dict(self.current_job)

    def prepare_dataset(self, dataset_name: str, class_names: List[str]) -> str:
        """Sets up folder structure and data.yaml for a custom dataset."""
        dataset_path = os.path.join(self.datasets_dir, dataset_name)
        os.makedirs(os.path.join(dataset_path, "images", "train"), exist_ok=True)
        os.makedirs(os.path.join(dataset_path, "images", "val"), exist_ok=True)
        os.makedirs(os.path.join(dataset_path, "labels", "train"), exist_ok=True)
        os.makedirs(os.path.join(dataset_path, "labels", "val"), exist_ok=True)

        data_yaml = {
            "path": dataset_path.replace("\\", "/"),
            "train": "images/train",
            "val": "images/val",
            "names": {i: name for i, name in enumerate(class_names)}
        }

        yaml_path = os.path.join(dataset_path, "data.yaml")
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(data_yaml, f, default_flow_style=False)

        return yaml_path

    def list_datasets(self) -> List[Dict[str, Any]]:
        """Lists existing datasets in the datasets directory."""
        datasets = []
        if os.path.exists(self.datasets_dir):
            for d in os.listdir(self.datasets_dir):
                d_path = os.path.join(self.datasets_dir, d)
                yaml_file = os.path.join(d_path, "data.yaml")
                if os.path.isdir(d_path) and os.path.exists(yaml_file):
                    try:
                        with open(yaml_file, "r", encoding="utf-8") as f:
                            data = yaml.safe_load(f)
                            train_imgs = len(os.listdir(os.path.join(d_path, "images", "train"))) if os.path.exists(os.path.join(d_path, "images", "train")) else 0
                            val_imgs = len(os.listdir(os.path.join(d_path, "images", "val"))) if os.path.exists(os.path.join(d_path, "images", "val")) else 0
                            datasets.append({
                                "name": d,
                                "classes": list(data.get("names", {}).values()) if isinstance(data.get("names"), dict) else data.get("names", []),
                                "total_images": train_imgs + val_imgs,
                                "train_images": train_imgs,
                                "val_images": val_imgs
                            })
                    except Exception:
                        pass
        return datasets

    def start_training(
        self,
        dataset_name: str,
        output_model_name: str,
        base_model: str = "general_yolo11n.pt",
        epochs: int = 10,
        imgsz: int = 416,
        batch_size: int = 8
    ) -> bool:
        """Starts asynchronous training of a custom YOLO model."""
        with self.lock:
            if self.is_training:
                return False

            self.is_training = True
            self.current_job = {
                "status": "starting",
                "progress": 0,
                "epoch": 0,
                "total_epochs": epochs,
                "loss": 0.0,
                "map50": 0.0,
                "log": [f"Initializing custom training for '{output_model_name}'..."],
                "error": None,
                "output_model": None
            }

        def _train_worker():
            try:
                dataset_path = os.path.join(self.datasets_dir, dataset_name)
                yaml_path = os.path.join(dataset_path, "data.yaml")
                if not os.path.exists(yaml_path):
                    raise FileNotFoundError(f"data.yaml not found at {yaml_path}")

                # Choose device
                device = 0 if torch.cuda.is_available() else "cpu"
                device_str = "NVIDIA GPU (CUDA)" if device == 0 else "CPU"
                
                with self.lock:
                    self.current_job["log"].append(f"Using compute device: {device_str}")
                    self.current_job["status"] = "training"

                # Base model path
                base_path = os.path.join(self.models_dir, base_model)
                if not os.path.exists(base_path):
                    base_path = "yolo11n.pt"

                model = YOLO(base_path)

                # Custom callback to capture training progress
                def on_fit_epoch_end(trainer):
                    with self.lock:
                        curr_epoch = trainer.epoch + 1
                        total = trainer.epochs
                        # Extract loss metrics
                        loss_items = getattr(trainer, "loss_items", None)
                        loss_val = float(sum(loss_items)) if loss_items is not None else 0.0
                        
                        # Extract mAP metrics if available
                        metrics = getattr(trainer, "metrics", {})
                        map50 = float(metrics.get("metrics/mAP50(B)", 0.0))

                        self.current_job["epoch"] = curr_epoch
                        self.current_job["total_epochs"] = total
                        self.current_job["progress"] = int((curr_epoch / total) * 100)
                        self.current_job["loss"] = round(loss_val, 4)
                        self.current_job["map50"] = round(map50, 4)
                        self.current_job["log"].append(
                            f"Epoch {curr_epoch}/{total} completed | Loss: {self.current_job['loss']} | mAP50: {self.current_job['map50']}"
                        )

                model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

                # Execute training
                results = model.train(
                    data=yaml_path,
                    epochs=epochs,
                    imgsz=imgsz,
                    batch=batch_size,
                    device=device,
                    workers=0,  # avoid multiprocessing issues on Windows
                    verbose=True
                )

                # Locate best.pt
                best_weight = getattr(model.trainer, "best", None)
                if not best_weight or not os.path.exists(str(best_weight)):
                    save_dir = getattr(model.trainer, "save_dir", None)
                    if save_dir:
                        best_weight = os.path.join(str(save_dir), "weights", "best.pt")

                dest_filename = f"{output_model_name}.pt" if not output_model_name.endswith(".pt") else output_model_name
                dest_path = os.path.join(self.models_dir, dest_filename)

                if best_weight and os.path.exists(str(best_weight)):
                    shutil.copy(str(best_weight), dest_path)
                    with self.lock:
                        self.current_job["status"] = "completed"
                        self.current_job["progress"] = 100
                        self.current_job["output_model"] = dest_filename
                        self.current_job["log"].append(f"Training succeeded! Model saved as {dest_filename}")
                else:
                    raise RuntimeError("Training finished but weights file 'best.pt' was not found.")

            except Exception as e:
                with self.lock:
                    self.current_job["status"] = "failed"
                    self.current_job["error"] = str(e)
                    self.current_job["log"].append(f"Training error: {str(e)}")
            finally:
                with self.lock:
                    self.is_training = False

        self.training_thread = threading.Thread(target=_train_worker, daemon=True)
        self.training_thread.start()
        return True
