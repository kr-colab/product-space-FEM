Here we will have tools to wrangle data for inference and crossvalidation.

Things in a class (`SpatialDivergenceData`):
- property: sample location names and coordinates
- property: pairwise divergences and diversities
- property: scaling factors applied to coordinates and divergences
- method: read info from csv files


Steps we need to do, for which functionality can go here:

- normalize spatial and genetic data
    * `SpatialDivergenceData.normalize( )`

- compute kernel density estimate on the boundary
    * `SpatialDivergenceData.boundary_fn( )`

- slice spatial *and* genetic data all at once intro train/test sets
    * `SpatialDivergenceData.split( )`

And, maybe:

- make and save a mesh XML file from:
    * a habitat raster
    * a cutoff value in the raster
    * a set of sampling locations
