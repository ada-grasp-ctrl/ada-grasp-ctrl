# python src/main.py setting=tabletop hand=shadow task=format exp_name=learn task.data_name=Learning task.max_num=100 task.data_path=../DexLearn/output/bodex_tabletop_shadow_nflow_debug0/tests/step_045000
python src/main.py setting=tabletop hand=shadow task=dummy_arm_qpos exp_name=learn task.max_num=-1
python src/main.py setting=tabletop hand=dummy_arm_shadow task=control_eval exp_name=learn task.method=ours task.input_data=grasp_dir task.debug_viewer=False
python src/main.py setting=tabletop hand=dummy_arm_shadow task=control_stat exp_name=learn task.method=ours
