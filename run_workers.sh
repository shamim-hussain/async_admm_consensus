if [ $1 = '--help' ]
then
echo run_workers.sh script num_worker port beta S tau steps num_gpu worker_range
else

script=$1
num_worker=${2:-2}
port=${3:-33333}
beta=${4:-"1.0"}
S=${5:-"$2"}
tau=${6:-1}
steps=${7:-10}
num_gpu=${8:-0}
worker_range=${9:-"0-$2"}

IFS='-' read from to <<<$worker_range

echo ----------------------------------

echo num_worker = $num_worker
echo beta = $beta
echo S = $S
echo tau = $tau
echo steps = $steps
echo num_gpu = $num_gpu
echo from = $from
echo to = $to
echo ----------------------------------


if [ $num_gpu -eq 0 ]
then
    if [ $to -eq $num_worker ]
    then
        echo Starting MASTER
        python $script $num_worker $num_worker $port --beta $beta --S $S --tau $tau --steps $steps --device cpu:0 &
        sleep 2

        for w_i in $(seq $from $((to-1)))
        do
            echo Starting WORKER:$w_i
            python $script $num_worker $w_i $port --beta $beta --S $S --tau $tau --steps $steps --device cpu:0 &
            sleep 0.5
        done

    else
        for w_i in $(seq $from $to)
        do
            echo Starting WORKER:$w_i
            python $script $num_worker $w_i $port --beta $beta --S $S --tau $tau --steps $steps --device cpu:0 &
            sleep 0.5
        done

    fi




else
    if [ $to -eq $num_worker ]
    then
        echo Starting MASTER
        python $script $num_worker $num_worker $port --beta $beta --S $S --tau $tau --steps $steps --device cuda:0 &
        sleep 2

        for w_i in $(seq $from $((to-1)))
        do
            echo Starting WORKER:$w_i
            python $script $num_worker $w_i $port --beta $beta --S $S --tau $tau --steps $steps --device cuda:$(expr $w_i % $num_gpu) &
            sleep 0.5
        done

    else
        for w_i in $(seq $from $to)
        do
            echo Starting WORKER:$w_i
            python $script $num_worker $w_i $port --beta $beta --S $S --tau $tau --steps $steps --device cuda:$(expr $w_i % $num_gpu) &
            sleep 0.5
        done

    fi
fi


jobs

fi