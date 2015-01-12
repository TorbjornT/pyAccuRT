'''Dict with default values for AccuRT.'''

defaults = {}

defaults['main'] = {
  "SOURCE_TYPE": "earth_solar",
  "SOURCE_SCALING_FACTOR": "1.0",
  "SOURCE_ZENITH_ANGLE": "45",
  "BOTTOM_BOUNDARY_SURFACE" : "loamy_sand",
  "BOTTOM_BOUNDARY_SURFACE_SCALING_FACTOR" : "1.0",
  "STREAM_UPPER_SLAB_SIZE" : "8",
  "STREAM_LOWER_SLAB_PARAMETERS" : "1.0 2.0",
  "LAYER_DEPTHS_UPPER_SLAB" : '''30.0e3 50.0e3 60.0e3 70.0e3 
                          76.0e3 80.0e3 84.0e3 88.0e3 
                          90.0e3 92.0e3 94.0e3 96.0e3 
                          98.0e3 100.0e3''',
  "LAYER_DEPTHS_LOWER_SLAB" : "100",
  "MATERIALS_INCLUDED_UPPER_SLAB" : "earth_atmospheric_gases aerosols",
  "MATERIALS_INCLUDED_LOWER_SLAB" : "pure_water water_impurity_ccrr",
  "DETECTOR_DEPTHS_UPPER_SLAB" : "0 99999.999",
  "DETECTOR_DEPTHS_LOWER_SLAB" : "0.0001 1 10 20",
  "DETECTOR_AZIMUTH_ANGLES" : "0:20:180",
  "DETECTOR_POLAR_ANGLES" : "0:2:180",
  "DETECTOR_WAVELENGTHS" : "400:50:700",
  "DETECTOR_WAVELENGTH_BAND_WIDTHS" : '''270 1.0  
	 	                  4000 1.0''',
  "SAVE_COSINE_IRRADIANCE" : "true",
  "SAVE_SINE_IRRADIANCE" : "false",
  "SAVE_SCALAR_IRRADIANCE" : "false",
  "SAVE_RADIANCE" : "false",
  "SAVE_IOPS" : "false",
  "SAVE_BOTTOM_BOUNDARY_SURFACE" : "false",
  "SAVE_MATERIAL_PROFILE" : "true",
  "PROFILE_OUTPUT_WAVELENGTH" : "500",
  "PRINT_PROGRESS_TO_SCREEN" : "true",
  "REPEATED_RUN_SIZE" : "1"
}


defaults['ice'] = {
"PROFILE_LABEL" : "layer_numbering",
"BRINE_PROFILE " : '''1 0.05
                 2 0.01''',
"BUBBLE_PROFILE" : '''1 0.01
                 2 0.005''',
"BRINE_EFFECTIVE_RADIUS" : '''1 100
                         2 150''',
"BUBBLE_EFFECTIVE_RADIUS" : '''1 100
                          2 200''',   
"IMPURITY_PROFILE" : '''1 1e-8
                   2 1e-7''',
"IMPURITY_IMAG" : "0.4",
"INTERNAL_MIXING" : "false",
"USE_HG_PHASE_FUNCTION_OVERRIDE" : "false",  
"USE_PARAMETERIZED_MIE_CODE" : "true",
"BRINE_SIZE_DISTRIBUTION_WIDTH" : '''1 0.1
                                2 0.2''',
"BUBBLE_SIZE_DISTRIBUTION_WIDTH" : '''1 0.1
                                 2 0.1''',
"QUADRATURE_POINTS" : "50"
}





def getdefaults():
    return defaults
