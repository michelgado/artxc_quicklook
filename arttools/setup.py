from distutils.core import setup, Extension
import numpy

m1 = Extension("src_rate_solvers", ["rate_solvers.c", "src_rate_solvers.c"], include_dirs = ["./", numpy.get_include()])
m2 = Extension("psf_functions", ["rate_solvers.c", "psf_functions.c"], include_dirs = ["./", numpy.get_include()])

def main():
    setup(name="lkl_solution",
          version="0.1",
          author="andrey",
          author_email="san@iki.rssi.ru",
          ext_modules=[m1, m2])

if __name__ == "__main__":
    main()
