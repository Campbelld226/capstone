import sys
import os
import re
import numpy as np
from scipy.signal import savgol_filter
from scipy.signal import gauss_spline
from matplotlib import pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import interp1d
# TODO: We should compile a list of questions for our meeting with Cartledge about what he expects from this program. \
#  - Does he just want command line prompts or a basic GUI?                                                           \
#  - What does he expect his input to look like?                                                                      \
#  - What does he expect the output to be?                                                                            \
#  - Does he have a filter system for signals already?

# TODO: Fix encoding issues in files. (need utf-8)

# TODO: Classification \
#  Find features of each point \
#  Label each point \


class Planet:
    def __init__(self, name):
        self.name = name
        self.data_array = []

    # Defines the string representation of the Planet class as the planet name
    def __repr__(self):
        return str(self.name)

    # Adds a Dataset object to the planet data array
    def add_dataset(self, filename):
        data = Dataset()
        data.parse_data(filename)
        self.data_array.append(data)


class Dataset:
    def __init__(self):
        self.name = None
        self.data_quality = None
        self.epoch_array = []
        self.depth_array = []
        self.error_array = []

    def __repr__(self):
        return str(self.name)

    # Parses a data file and populates the Dataset class attributes
    def parse_data(self, filename):
        self.name = filename
        # TODO: Figure out how to make pycharm ignore specific style issues. It feels the escaping of these periods \
        #  is a style issue and I want it to ignore this specific case but not any other styling issues.
        # Gets the data quality from the filename
        self.data_quality = re.findall("\d\.\d*[^\.]", filename)[0][0]
        for line in open(filename, 'r'):
            # If the line doesn't start with a digit discard it
            if not line[0].isdigit():
                continue
            # TODO: Currently this is getting around different data formats with an ugly try-except and adding any \
            #  extra info into the error field. This is gross and I hate it
            try:
                # Unpacks the array returned from the split function into temp variables
                # The * in *error allows for a variable number of values to be saved in error as an array
                epoch, depth, *error = line.split()

            # if < 3 values in line set error to None
            except ValueError:
                epoch, depth = line.split()
                error = None
            epoch = float(epoch) - int(float(epoch))
            self.epoch_array.append(epoch)
            self.depth_array.append(float(depth))
            self.error_array.append(error)


def adjust_boundaries(min, max, percent):

    min -= (min*percent)
    max += (max*percent)

    return min, max


# TODO: Begin trying different SciPy things on this data and seeing what comes out
if __name__ == '__main__':
    planet_array = []
    # Iterate through each file in transit_data
    # for subdir, dirs, files in os.walk('transit_data'):
    #     try:
    #         # The main directory (transit_data) has no subdir and will throw an exception when trying to index after
    #         # the split. This is fine since we want to skip it anyways
    #         planet_name = subdir.split('/')[1]
    #         planet = Planet(planet_name)
    #         for file in files:
    #             file.encode("utf-8")
    #             # add each file within a planets directory as a dataset
    #             planet.add_dataset(os.path.join(subdir, file))
    #             # print(os.path.join(subdir, file))
    #         planet_array.append(planet)
    #     except IndexError:
    #         continue

    directory = sys.argv[1]
    planet_name = directory.split('/')[2].split('_')[0]
    print(planet_name, directory)
    planet = Planet(planet_name)
    planet.add_dataset(directory)
    planet_array.append(planet)

    # Test prints
    print("Planet Name: ", planet_array[0])
    print("File: ", planet_array[0].data_array[0])
    print("Epoch Time[0]: ", planet_array[0].data_array[0].epoch_array[0])
    print("Depth of Transit[0]: ", planet_array[0].data_array[0].depth_array[0])
    print("Error/Other-Data[0]: ", planet_array[0].data_array[0].error_array[0])
    print("Data Quality: ", planet_array[0].data_array[0].data_quality)

    # Raw data
    x = planet_array[0].data_array[0].epoch_array
    y = planet_array[0].data_array[0].depth_array

    minimum = np.amin(planet_array[0].data_array[0].depth_array).item()
    maximum = np.amax(planet_array[0].data_array[0].depth_array).item()
    print(minimum, maximum)
    # minimum, maximum = adjust_boundaries(minimum, maximum, 0.05)
    minimum, maximum = adjust_boundaries(-2, 2, 0.05)
    print(minimum, maximum)
    plt.plot(x, y, 'o')
    # print(planet_array[0].data_array[0].epoch_array)

    # Savgol filter
    smooth_x = savgol_filter(x, 37, 2)
    smooth_y = savgol_filter(y, 37, 2)
    plt.plot(smooth_x, smooth_y, "o")

    # Univariate spline
    # TODO: Splines are not it
    s = UnivariateSpline(smooth_x, smooth_y, s=1)
    ys = s(smooth_x)
    plt.plot(smooth_x, ys)

    # Interpolation
    fs = interp1d(smooth_x, smooth_y, kind='cubic', fill_value='extrapolate')
    plt.plot(x, fs(x), '-')

    # TODO: Dynamically set the y limits based on the data
    # Set the y limits to get the proper shape
    plt.ylim(minimum, maximum)
    plt.xlabel("Epoch")
    plt.ylabel("Depth")
    plt.show()
