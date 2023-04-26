# assumes python setup_crossval_grid.py has already been executed
# example command: snakemake -C base_name=simulation/density_saddle/out_2877096093782_stats/rep349832 --profile ~/.config/snakemake/talapas/

base_name = 'not_given_in_config'
if 'base_name' in config:
    base_name = config['base_name'] # can set with --config base_name="..."
seed = 123
num_samples = 900

if 'seed' in config:
    seed = config['seed']
if 'num_samples' in config:
    num_samples = config['num_samples']
outs, = glob_wildcards(base_name + "_{iter}/xval_params.json")

rule all:
    input:
        [f"{base_name}_{o}/solutions.png" for o in outs] + [f"{base_name}_{o}/results.crossvalidation_results.html" for o in outs]

rule summarise_single_xval:
    input:
        base_name + "_{o}/results.pkl"
    output:
        base_name + "_{o}/results.crossvalidation_results.html"
    shell:
        """
            ./crossvalidation_results.sh {input}
        """

rule plot_xval:
    input:
        base_name + "_{o}/results.pkl"
    output:
        base_name + "_{o}/solutions.png"
    shell:
        """
            python plot_crossvalidation.py {input}
        """

rule cross_val:
    input:
        base_name + "_{o}/xval_params.json"
    output:
        base_name + "_{o}/results.pkl"
    resources:
        runtime = 720,
        mem_mb = 15000
    shell:
        """
            python crossvalidation.py --json {input}
        """

rule stats:
    input:
        base_name + ".trees"
    output:
        [base_name + "_stats/rep{seed}.pairstats.csv",
         base_name + "_stats/rep{seed}.stats.csv"]
    resources:
        runtime = 720,
        mem_mb = 6000
    shell:
        """
            python compute_stats.py {input} {num_samples} {seed}
        """
