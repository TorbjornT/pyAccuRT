import numpy as np


def read_irradiance(filename):
    '''Read output irradiance textfiles from AccuRT model.
    Returns number of runs, streams, detector depths and wavelengths,
    and numpy arrays of depths, wavelengths and irradiance'''

    with open(filename, 'r') as infile:

        # read number of runs, streams, depths, wavelengths
        # and lists of detector depths, wavelengths
        nruns = int(infile.readline())
        nstreams = int(infile.readline())
        ndepths, nwavelengths = [int(j) for j in infile.readline().split()]
        depths = np.array([float(j) for j in infile.readline().split()])
        wavelengths = np.array([float(j) for j
                                in infile.readline().split()])

        # initiate array for irradiances
        irradiances = np.empty((ndepths, nwavelengths, nruns))

        # read values for first run
        for j in range(ndepths):
            irradiances[j, :, 0] = \
                [float(n) for n in infile.readline().split()]

        # read values for rest of runs
        for i in range(1, nruns):
            # skip lines with nstreams, ndepths, etc.
            for _ in range(4):
                next(infile)
            # read values
            for j in range(ndepths):
                irradiances[j, :, i] = \
                    [float(n) for n in infile.readline().split()]

    return nruns, nstreams, ndepths, nwavelengths, depths,\
        wavelengths, irradiances


def read_radiance(filename):
    '''Read output radiance.txt from AccuRT.
    Returns number of runs, streams, detector depths and wavelengths,
    and numpy arrays of dephts, wavelengths, polar and azimuth angles
    and radiance.

    Dimensions of radiance array are
    (depth) x (wavelength) x (polar angle) x (azimuth angle) x (run number)
    '''

    with open(filename, 'r') as infile:
        nruns = int(infile.readline())
        nstreams = int(infile.readline())
        ndepths, nwavelengths, npolarangles, nazimuthangles = \
            [int(j) for j in infile.readline().split()]
        depths = np.array([float(j) for j in infile.readline().split()])
        wavelengths = np.array([float(j) for j
                                in infile.readline().split()])
        polarangles = np.array([float(j) for j
                                in infile.readline().split()])
        azimuthangles = np.array([float(j) for j
                                  in infile.readline().split()])

        radiances = np.empty((ndepths, nwavelengths, npolarangles,
                              nazimuthangles, nruns))

        rad = np.array([float(j) for j in infile.readline().split()])
        radiances[:, :, :, :, 0] = rad.reshape(ndepths,
                                               nwavelengths,
                                               npolarangles,
                                               nazimuthangles)

        for i in range(1, nruns):
            # skip lines with nstreams, ndepths, etc.
            for _ in range(6):
                next(infile)
            rad = np.array([float(j) for j in infile.readline().split()])
            # read values
            radiances[:, :, :, :, i] = rad.reshape(ndepths,
                                                   nwavelengths,
                                                   npolarangles,
                                                   nazimuthangles)

    return radiances, polarangles, azimuthangles, nruns, nstreams,\
        ndepths, nwavelengths, depths, wavelengths


def read_iops(filename):
    '''Read iops.txt, returns dict.'''

    with open(filename, 'r') as infile:
        nruns = int(infile.readline())

        total_optical_depth = []
        absorption_coefficients = []
        scattering_coefficients = []
        scattering_scaling_factors = []
        phase_moments = []
        layer_depths = []
        wavelengths = []

        for _ in range(nruns):
            nlayerdepths, nwavelengths, nphasemoments = \
                [int(x) for x in infile.readline().split()]

            layer_depths.append(np.array([float(x) for x in
                                          infile.readline().split()]))
            wavelengths.append(np.array([float(x) for x in
                                         infile.readline().split()]))

            _ToD = np.empty((nlayerdepths, nwavelengths))
            _AC = np.empty((nlayerdepths, nwavelengths))
            _SC = np.empty((nlayerdepths, nwavelengths))
            _SSF = np.empty((nlayerdepths, nwavelengths))
            _PM = np.empty((nlayerdepths, nwavelengths, nphasemoments))

            for j in range(nlayerdepths):
                for k in range(nwavelengths):
                    line = infile.readline().split()
                    _ToD[j, k] = float(line.pop(0))
                    _AC[j, k] = float(line.pop(0))
                    _SC[j, k] = float(line.pop(0))
                    _SSF[j, k] = float(line.pop(0))
                    _PM[j, k, :] = np.array(line, dtype='float')

            total_optical_depth.append(_ToD.copy())
            absorption_coefficients.append(_AC.copy())
            scattering_coefficients.append(_SC.copy())
            scattering_scaling_factors.append(_SSF.copy())
            phase_moments.append(_PM.copy())

        if nruns > 1:
            npm = phase_moments[0].shape[-1]
            eq_npm = True
            for i in range(1,len(phase_moments)):
                if phase_moments[i].shape[-1] != npm:
                    eq_npm = False
                    break
            if eq_npm:
                phase_moments = np.array(phase_moments)
        else:
            phase_moments = np.array(phase_moments)


        iops = dict(nruns=nruns,
                    layer_depths=np.array(layer_depths),
                    wavelengths=np.array(wavelengths)[0],
                    total_optical_depth=np.array(total_optical_depth),
                    absorption_coefficients=np.array(
                        absorption_coefficients),
                    scattering_coefficients=np.array(
                        scattering_coefficients),
                    scattering_scaling_factors=np.array(
                        scattering_scaling_factors),
                    phase_moments=phase_moments)

        return iops


def read_material_profile(filename):
    '''Read material_profile.txt'''
    material_profile = []
    with open(filename) as material_file:
        while True:
            line = material_file.readline()
            if line.startswith('runNo'):
                run = int(line.split()[1])
                break
        endoffile = False
        while True:
            if endoffile:
                break
            layer = -1
            material_profile.append([])
            while True:
                line = material_file.readline()
                if line.startswith('=') or \
                   line.startswith('-') or \
                   line.startswith('~'):
                    pass
                elif len(line) == 0:
                    endoffile = True
                    break
                elif line.startswith('runNo'):
                    run = int(line.split()[1])
                    break
                elif line.startswith('Layer '):
                    layer += 1
                    material_profile[run].append(dict())
                elif line.startswith('Bottom depth'):
                    z = float(line.split()[-2])
                    material_profile[run][layer]['bottomdepth'] = z
                else:
                    matname = line.replace(' ', '').strip()
                    material = dict()
                    conc, conctype = material_file.readline().split()
                    tau, ssa, g = [float(x) for x in
                                   material_file.readline().split()]
                    atau, btau = [float(x) for x in
                                  material_file.readline().split()]
                    a, b = [float(x) for x in
                            material_file.readline().split()]
                    df = float(material_file.readline())

                    material['concentration'] = float(conc)
                    material['concentrationtype'] = conctype[1:-1]
                    material['opticaldepth'] = tau
                    material['singlescatteringalbedo'] = ssa
                    material['asymmetryfactor'] = g
                    material['absorptionoptdep'] = atau
                    material['scatteringoptdep'] = btau
                    material['absorption'] = a
                    material['scattering'] = b
                    material['deltafitscalingfactor'] = df

                    material_profile[run][layer][matname] = material

    return material_profile
