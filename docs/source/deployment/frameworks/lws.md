(deployment-lws)=

# LWS

LeaderWorkerSet (LWS) is a Kubernetes API that aims to address common deployment patterns of AI/ML inference workloads.
A major use case is for multi-host/multi-node distributed inference.

vLLM can be deployed with [LWS](https://github.com/kubernetes-sigs/lws) on Kubernetes for distributed model serving.

Please see [this guide](https://github.com/kubernetes-sigs/lws/tree/main/docs/examples/vllm) for more details on
deploying vLLM on Kubernetes using LWS.
