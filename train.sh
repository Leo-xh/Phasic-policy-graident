RCALL_LOGDIR="./rcall_log" mpiexec -np 4 python -m phasic_policy_gradient.train --log_dir="./log/ppg-no-restore" --restore=False

RCALL_LOGDIR="./rcall_log" mpiexec -np 4 python -m phasic_policy_gradient.train --log_dir="./log/ppg-restore-no-aux-vf" --restore=True --use_aux_vf=False 

RCALL_LOGDIR="./rcall_log" mpiexec -np 4 python -m phasic_policy_gradient.train --log_dir="./log/ppg" --restore=True

RCALL_LOGDIR="./rcall_log" mpiexec -np 4 python -m phasic_policy_gradient.train --log_dir="./log/ppo-restore-1" --restore=True --use_aux_vf=False --n_epoch_pi 1 --n_epoch_vf 1 --n_aux_epochs 2 --arch shared

RCALL_LOGDIR="./rcall_log" mpiexec -np 4 python -m phasic_policy_gradient.train --log_dir="./log/ppo-restore-2" --restore=True --use_aux_vf=False --n_epoch_pi 1 --n_epoch_vf 1 --n_aux_epochs 4 --arch shared

RCALL_LOGDIR="./rcall_log" mpiexec -np 4 python -m phasic_policy_gradient.train --log_dir="./log/ppo-restore-3" --restore=True --use_aux_vf=False --n_epoch_pi 1 --n_epoch_vf 1 --n_aux_epochs 6 --arch shared

RCALL_LOGDIR="./rcall_log" mpiexec -np 4 python -m phasic_policy_gradient.train --n_epoch_pi 3 --n_epoch_vf 3 --n_aux_epochs 0 --arch shared --log_dir="./log/ppo"