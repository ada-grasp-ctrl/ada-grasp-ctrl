# python src/main.py setting=tabletop hand=leap_tac3d task=format exp_name=learn_2 task.data_name=Learning task.max_num=100 task.data_path=../DexLearn/output/bodex_tabletop_leap_tac3d_nflow_debug0/tests/step_050000
python src/main.py setting=tabletop hand=leap_tac3d task=dummy_arm_qpos exp_name=learn_2 task.max_num=-1
python src/main.py setting=tabletop hand=dummy_arm_leap_tac3d task=control_eval exp_name=learn task.method=ours task.input_data=grasp_dir task.debug_viewer=False
python src/main.py setting=tabletop hand=dummy_arm_leap_tac3d task=control_stat exp_name=learn task.method=ours