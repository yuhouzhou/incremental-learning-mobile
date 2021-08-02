# Baseline
python3 demo_baseline.py
# iCaRL
python3 -minclearn --options options/icarl/icarl_cifar100_mobilenetv2.yaml options/data/cifar100_3orders.yaml --initial-increment 50 --increment 1 --fixed-memory --device 0 --label icarl_cifar100_50steps_mobilenet --data-path data;
python3 -minclearn --options options/icarl/icarl_cifar100_mobilenetv2.yaml options/data/cifar100_3orders.yaml --initial-increment 50 --increment 2 --fixed-memory --device 0 --label icarl_cifar100_25steps_mobilenet --data-path data;
python3 -minclearn --options options/icarl/icarl_cifar100_mobilenetv2.yaml options/data/cifar100_3orders.yaml --initial-increment 50 --increment 5 --fixed-memory --device 0 --label icarl_cifar100_10steps_mobilenet --data-path data;
python3 -minclearn --options options/icarl/icarl_cifar100_mobilenetv2.yaml options/data/cifar100_3orders.yaml --initial-increment 50 --increment 10 --fixed-memory --device 0 --label icarl_cifar100_5steps_mobilenet --data-path data;
# BiC
python3 -minclearn --options options/bic/bic_cifar100_mobilenetv2.yaml options/data/cifar100_3orders.yaml --initial-increment 50 --increment 1 --fixed-memory --device 0 --label bic_cifar100_50steps_mobilenet --data-path data;
python3 -minclearn --options options/bic/bic_cifar100_mobilenetv2.yaml options/data/cifar100_3orders.yaml --initial-increment 50 --increment 2 --fixed-memory --device 0 --label bic_cifar100_25steps_mobilenet --data-path data;
python3 -minclearn --options options/bic/bic_cifar100_mobilenetv2.yaml options/data/cifar100_3orders.yaml --initial-increment 50 --increment 5 --fixed-memory --device 0 --label bic_cifar100_10steps_mobilenet --data-path data;
python3 -minclearn --options options/bic/bic_cifar100_mobilenetv2.yaml options/data/cifar100_3orders.yaml --initial-increment 50 --increment 10 --fixed-memory --device 0 --label bic_cifar100_5steps_mobilenet --data-path data;
# PODNet
python3 -minclearn --options options/podnet/podnet_cnn_cifar100_mobilenetv2.yaml options/data/cifar100_3orders.yaml --initial-increment 50 --increment 1 --fixed-memory --device 0 --label podnet_cnn_cifar100_50steps_mobilenet --data-path data;
python3 -minclearn --options options/podnet/podnet_cnn_cifar100_mobilenetv2.yaml options/data/cifar100_3orders.yaml --initial-increment 50 --increment 2 --fixed-memory --device 0 --label podnet_cnn_cifar100_25steps_mobilenet --data-path data;
python3 -minclearn --options options/podnet/podnet_cnn_cifar100_mobilenetv2.yaml options/data/cifar100_3orders.yaml --initial-increment 50 --increment 5 --fixed-memory --device 0 --label podnet_cnn_cifar100_10steps_mobilenet --data-path data;
python3 -minclearn --options options/podnet/podnet_cnn_cifar100_mobilenetv2.yaml options/data/cifar100_3orders.yaml --initial-increment 50 --increment 10 --fixed-memory --device 0 --label podnet_cnn_cifar100_5steps_mobilenet --data-path data;
