# Kubeflow Pipelines Setup Guide (vCluster-based)

This directory contains the configurations and instructions for deploying and running Kubeflow Pipelines (KFP) locally using **vCluster** with the Docker driver (also known as vīnd).

---

## 1. Prerequisites

Ensure you have the following installed on your host system:
* **Docker**
* **vCluster CLI** (v0.35.0+)
* **kubectl** (v1.30.0+)
* **Helm** (v3.10.0+)

---

## 2. Virtual Cluster Initialization

If you do not have an existing host Kubernetes cluster, you can run vCluster directly on Docker. This runs the Kubernetes control plane in isolated Docker containers, creating a lightweight, local Kubernetes environment.

### Step 2.1: Configure vCluster to use Docker driver
```bash
vcluster use driver docker
```

### Step 2.2: Create and start the virtual cluster
```bash
vcluster create churn-pipeline-cluster
```

### Step 2.3: Connect to the virtual cluster
This command sets up port forwarding and modifies your local `kubeconfig` to point to the virtual cluster:
```bash
vcluster connect churn-pipeline-cluster
```

Verify that you are connected and that the nodes are ready:
```bash
kubectl get nodes
```

---

## 3. Deploying Kubeflow Pipelines (KFP)

We will install KFP Standalone (platform-agnostic) on our virtual cluster.

### Step 3.1: Deploy KFP manifests
Run the following to apply the platform-agnostic KFP manifests to the cluster:
```bash
export PIPELINE_VERSION=2.2.0
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic-emissary?ref=$PIPELINE_VERSION"
```

### Step 3.2: Verify deployment
Wait for all KFP pods in the `kubeflow` namespace to become ready:
```bash
kubectl get pods -n kubeflow --watch
```

### Step 3.3: Port-forward KFP Dashboard
To access the Kubeflow Pipelines UI, port-forward the KFP UI service to your localhost:
```bash
kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80
```
Open your browser and navigate to `http://localhost:8080` to see the dashboard.

---

## 4. Pipeline Compilation & Run

The Python scripts in this folder define the component tasks and compile them into a pipeline specification.

### Step 4.1: Compile pipeline
Compile the `pipeline.py` script into a YAML DAG file:
```bash
.venv/bin/python kubeflow/pipeline.py
```
This generates `kubeflow/pipeline.yaml`.

### Step 4.2: Upload & Run
1. Go to the KFP UI at `http://localhost:8080`.
2. Click **Pipelines** -> **Upload Pipeline**.
3. Upload `kubeflow/pipeline.yaml`.
4. Create an **Experiment** and trigger a **Run**.
