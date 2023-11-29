import subprocess
classifier_choice = input("Select classifier (mnn, dt, ppr): ")

if classifier_choice == "mnn":
    subprocess.run(['python', 'phase_3_task_3_mnn.py'])
elif classifier_choice == "dt":
    subprocess.run(['python', 'phase_3_task_3_dt.py'])
elif classifier_choice == "ppr":
    subprocess.run(['python', 'ppr_cleaned.py'])
else:
    raise ValueError("Invalid classifier choice")
