#include "root_io.hpp"

#include <iostream>
#include <stdexcept>

#include <TFile.h>
#include <TH2.h>

Image load_image_root(const std::string& filename, const std::string& hist_name)
{
    TFile file(filename.c_str(), "READ");

    if (file.IsZombie())
        throw std::runtime_error("Could not open ROOT file: " + filename);

    TH2* hist = dynamic_cast<TH2*>(file.Get(hist_name.c_str()));

    if (!hist)
        throw std::runtime_error("Could not find TH2 '" + hist_name + "' in file: " + filename);

    int nx = hist->GetNbinsX();
    int ny = hist->GetNbinsY();

    Image image(ny, nx);

    for (int ix = 0; ix < nx; ++ix)
    {
        for (int iy = 0; iy < ny; ++iy)
        {
            image.data[ix * ny + iy] = static_cast<int>(
                hist->GetBinContent(ix + 1, iy + 1)
            );
        }
    }

    return image;
}

FloatImage load_pedestal_root(const std::string& filename, const std::string& hist_name)
{
    TFile file(filename.c_str(), "READ");

    if (file.IsZombie())
        throw std::runtime_error("Could not open ROOT file: " + filename);

    TH2* hist = dynamic_cast<TH2*>(file.Get(hist_name.c_str()));

    if (!hist)
        throw std::runtime_error("Could not find TH2 '" + hist_name + "' in file: " + filename);

    int nx = hist->GetNbinsX();
    int ny = hist->GetNbinsY();

    std::cout << "ROOT pedestal: nx = " << nx << ", ny = " << ny << std::endl;

    FloatImage pedestal(nx, ny);

    for (int ix = 0; ix < nx; ++ix)
    {
        for (int iy = 0; iy < ny; ++iy)
        {
            pedestal.data[iy * nx + ix] = static_cast<float>(
                hist->GetBinContent(ix + 1, iy + 1)
            );
        }
    }

    return pedestal;
}