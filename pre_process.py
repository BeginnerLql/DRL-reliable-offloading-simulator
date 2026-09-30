# pre_process.py
"""
Pre-process launcher (run from project root).
Generates input Excel files into data/:
- server_info.xlsx
- task_parameters.xlsx
"""

def main():
    from tools.prepare_eua_topology import prepare_topology
    from tools.generate_server_and_task_parameters import main as gen_main

    prepare_topology()
    gen_main()


if __name__ == "__main__":
    main()
