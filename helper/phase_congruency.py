import phasepack as pp
from skimage import img_as_float
from skimage.util.arraycrop import crop
import numpy as np
# Returns:
# M: Maximum moment of phase congruency covariance, which can be used as a measure of edge strength
# m: Minimum moment of phase congruency covariance, which can be used as a measure of corner strength
# ori: Orientation image, in integer degrees (0-180), positive angles anti-clockwise.
# ft: Local weighted mean phase angle at every point in the image. A value of pi/2 corresponds to a bright line, 0 to a step and -pi/2 to a dark line.
# PC: A list of phase congruency images (values between 0 and 1), one per orientation.
# EO: A list containing the complex-valued convolution results.
# T: Calculated noise threshold (can be useful for diagnosing noise characteristics of images). Once you know this you can then specify fixed thresholds and save some computation time.

# Assumption: Most use from (M + m), PC.sum() and ft
# (M + m) emphasizes corners and edges
# PC seems to emphasize corners and edges, similar to M and m  but with wider range
# ft seems to emphasize smooth regions
def phase_congruency(img, nscale=5, norient=6, minWaveLength=3, mult=2.1, sigmaOnf=0.55, k=2., cutOff=0.5, g=10., noiseMethod=-1):
        img = img.copy()
        # Pad to avoid border effects
        img = np.pad(img, nscale, mode="constant", constant_values=0)
        
        # Description of parameters in the original package
        M, m, ori, ft, PC, EO, T = pp.phasecong(img, 
                nscale=nscale, 
                norient=norient, 
                minWaveLength=minWaveLength, 
                mult=mult, 
                sigmaOnf=sigmaOnf, 
                k=k, 
                cutOff=cutOff, 
                g=g, 
                noiseMethod=noiseMethod)
        
        # Remove padding from result
        M = crop(M, nscale)
        m = crop(m, nscale)
        ori = crop(ori, nscale)
        ft = crop(ft, nscale)
        for idx, im in enumerate(PC):
                PC[idx] = crop(im, nscale)
        for idx, li in enumerate(EO):
                for idx2, im in enumerate(li):
                        EO[idx][idx2] = crop(im, nscale)
        
        return M, m, ori, ft, PC, EO, T


# Returns:
# phaseSym: Phase symmetry image (values between 0 and 1).
# orientation: Orientation image. Orientation in which local symmetry energy is a maximum, in degrees (0-180), angles positive anti-clockwise. Note that the orientation info is quantized by the number of orientations
# totalEnergy: Un-normalised raw symmetry energy which may be more to your liking.
# T: Calculated noise threshold (can be useful for diagnosing noise characteristics of images). Once you know this you can then specify fixed thresholds and save some computation time.
def phase_symmetry(img, nscale=5, norient=6, minWaveLength=3, mult=2.1, sigmaOnf=0.55, k=2., polarity=0, noiseMethod=-1):
        img = img.copy()
        # Pad to avoid border effects
        img = np.pad(img, nscale, mode="constant", constant_values=0)
        # Description of parameters in the original package
        phaseSym, orientation, totalEnergy, T = pp.phasesym(img, 
                nscale=nscale, 
                norient=norient, 
                minWaveLength=minWaveLength, 
                mult=mult, 
                sigmaOnf=sigmaOnf, 
                k=k, 
                polarity=polarity, 
                noiseMethod=noiseMethod)
        
        # Remove padding from result
        phaseSym = crop(phaseSym, nscale)
        orientation = crop(orientation, nscale)
        totalEnergy = crop(totalEnergy, nscale)

        return phaseSym, orientation, totalEnergy, T