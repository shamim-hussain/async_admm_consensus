script=$1
num_worker=$2
beta=${3:-1.0}
S=${4:-$1}
tau=${5:-1}
steps=${6:-50}



echo ----------------------------------

echo num_worker = $num_worker
echo beta = $beta
echo S = $S 
echo tau = $tau
echo steps = $steps

echo ----------------------------------

echo Generating knownhosts.json
python knownhosts_gen.py $num_worker
sleep 1

echo Starting MASTER
python script $num_worker --beta $beta --S $S --tau $tau --steps $steps --device cuda:$num_worker &
sleep 1

for w_i in $(seq 0 $(expr $num_worker - 1))
do
    echo Starting WORKER:$w_i
    python script $w_i --beta $beta --S $S --tau $tau --steps $steps --device cuda:$w_i &
    sleep 1
done

jobs

