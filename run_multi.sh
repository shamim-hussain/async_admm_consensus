script=$1
num_worker=$2
port=$3
beta=${4:-1.0}
S=${5:-$2}
tau=${6:-1}
steps=${7:-50}
num_device=${8:-6}


echo ----------------------------------

echo num_worker = $num_worker
echo beta = $beta
echo S = $S 
echo tau = $tau
echo steps = $steps

echo ----------------------------------

echo Starting MASTER
python $script $num_worker $num_worker $port --beta $beta --S $S --tau $tau --steps $steps --device cuda:0 &
sleep 2

for w_i in $(seq 0 $(expr $num_worker - 1))
do
    echo Starting WORKER:$w_i
    python $script $num_worker $w_i $port --beta $beta --S $S --tau $tau --steps $steps --device cuda:$(expr $w_i % $num_device) &
    sleep 0.5
done

jobs

