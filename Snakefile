# assumes python setup_crossval_grid.py has already been executed
# example command: snakemake -c6  -C base_name=simulation/density_saddle/out_2877096093782_stats/rep349832 --profile ~/.config/snakemake/talapas/

base_name = config['base_name'] # can set with --config base_name="..."
outs, = glob_wildcards(base_name + "_{iter}/xval_params.json")

rule all:
    input:
        [f"{base_name}_{o}/results.pkl" for o in outs]

rule cross_vall:
    input:
        base_name + "_{o}/xval_params.json"
    output:
        base_name + "_{o}/results.pkl"
    shell:
        """
            python crossvalidation.py {input}
        """
