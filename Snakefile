# assumes python setup_crossval_grid.py has already been executed
# example command: snakemake -c6  -C base_dir=simulation/density_saddle/out_2877096093782_stats

base_dir = config['base_dir'] # can set with --config base_dir="..."
outs, = glob_wildcards(base_dir + "/xval_{iter}/xval_params.json")

rule all:
    input:
        [f"{base_dir}/xval_{o}/results.pkl" for o in outs]

rule cross_vall:
    input:
        base_dir + "/xval_{o}/xval_params.json"
    output:
        base_dir + "/xval_{o}/results.pkl"
    shell:
        """
            python crossvalidation.py {input}
        """