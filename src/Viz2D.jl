module Viz2D
# -----------------------------------------------------------------------------
# 2D visualization (n = 2) for Z^2 flanges:
#   - Regions for indecomposable flats (blue) and injectives (red) inside a box
#   - Constant-subdivision heatmap dim(M_g) on Z^2
#
# CairoMakie is OPTIONAL. If it is not available in the active environment,
# this module still loads, but the plotting functions throw a helpful error.
# -----------------------------------------------------------------------------

import ..FlangeZn: Flange, active_flats, active_injectives
using ..CoreModules: QQ
import ..ExactQQ: rankQQ

const HAVE_CAIROMAKIE = try
    @eval import CairoMakie
    true
catch
    false
end

export HAVE_CAIROMAKIE, draw_flange_regions2D, draw_constant_subdivision2D

# -----------------------------------------------------------------------------
# Helper for optional dependency behavior
# -----------------------------------------------------------------------------
function _viz2d_missing_dep()
    error("Viz2D requires CairoMakie.jl. Install it in the active environment to use draw_flange_regions2D/draw_constant_subdivision2D.")
end

if HAVE_CAIROMAKIE

    # Use a stable point type name across Makie versions.
    const _Point2 = isdefined(CairoMakie, :Point2f0) ? CairoMakie.Point2f0 : CairoMakie.Point2f

    "Rectangle polygon clipped to [ell,u] with optional axis bounds."
    function _clip_rect_2d(ell::NTuple{2,Float64}, u::NTuple{2,Float64};
                           L1=-Inf, U1=+Inf, L2=-Inf, U2=+Inf)
        xlo = max(ell[1], L1); xhi = min(u[1], U1)
        ylo = max(ell[2], L2); yhi = min(u[2], U2)
        if xlo <= xhi && ylo <= yhi
            return _Point2[(xlo, ylo), (xhi, ylo), (xhi, yhi), (xlo, yhi)]
        else
            return _Point2[]
        end
    end

    """
        draw_flange_regions2D(fig, ax, FG; box=(ell,u), alpha_up=0.25, alpha_dn=0.25)

    Draw regions for flats (blue) and injectives (red) in the box `box = (ell,u)`.

    Notes:
    - This is intended only for flanges in ambient dimension n = 2.
    - Requires CairoMakie.
    """
    function draw_flange_regions2D(fig, ax, FG::Flange;
                                   box=(Float64[0,0], Float64[5,5]),
                                   alpha_up=0.25, alpha_dn=0.25)
        FG.n == 2 || error("draw_flange_regions2D only supports n = 2 flanges.")

        ell = (float(box[1][1]), float(box[1][2]))
        u   = (float(box[2][1]), float(box[2][2]))

        # Box outline + grid
        CairoMakie.lines!(ax,
            [ell[1], u[1], u[1], ell[1], ell[1]],
            [ell[2], ell[2], u[2], u[2], ell[2]];
            linewidth=1)

        for x in ceil(Int, ell[1]):floor(Int, u[1])
            CairoMakie.lines!(ax, [x, x], [ell[2], u[2]]; linewidth=0.4, color=:gray, alpha=0.25)
        end
        for y in ceil(Int, ell[2]):floor(Int, u[2])
            CairoMakie.lines!(ax, [ell[1], u[1]], [y, y]; linewidth=0.4, color=:gray, alpha=0.25)
        end

        # Flats (blue): x >= b on non-free coordinates
        for (j, F) in enumerate(FG.flats)
            L1 = F.tau.coords[1] ? -Inf : float(F.b[1])
            L2 = F.tau.coords[2] ? -Inf : float(F.b[2])
            poly = _clip_rect_2d(ell, u; L1=L1, L2=L2)
            if !isempty(poly)
                CairoMakie.poly!(ax, poly; color=(:dodgerblue, alpha_up), strokewidth=0)
                # Label near the lower-left corner
                CairoMakie.text!(ax, "U$(j)";
                    position=(poly[1][1] + 0.1, poly[1][2] + 0.1),
                    textsize=10, color=:navy)
            end
        end

        # Injectives (red): x <= b on non-free coordinates
        for (i, E) in enumerate(FG.injectives)
            U1 = E.tau.coords[1] ? +Inf : float(E.b[1])
            U2 = E.tau.coords[2] ? +Inf : float(E.b[2])
            poly = _clip_rect_2d(ell, u; U1=U1, U2=U2)
            if !isempty(poly)
                CairoMakie.poly!(ax, poly; color=(:crimson, alpha_dn), strokewidth=0)
                # Label near the upper-right corner
                CairoMakie.text!(ax, "D$(i)";
                    position=(poly[3][1] - 0.6, poly[3][2] - 0.6),
                    textsize=10, color=:darkred)
            end
        end

        CairoMakie.hidespines!(ax)
        ax.xlabel = "x1"
        ax.ylabel = "x2"
        CairoMakie.xlims!(ax, ell[1], u[1])
        CairoMakie.ylims!(ax, ell[2], u[2])
        return fig, ax
    end

    """
        draw_constant_subdivision2D(fig, ax, FG; box=(ell,u))

    Heatmap with each unit cell [x1,x1+1] \\times [x2,x2+1] colored by dim(M_g) at g=(x1,x2).

    Notes:
    - This is intended only for flanges in ambient dimension n = 2.
    - Requires CairoMakie.
    """
    function draw_constant_subdivision2D(fig, ax, FG::Flange;
                                         box=(Float64[0,0], Float64[5,5]))
        FG.n == 2 || error("draw_constant_subdivision2D only supports n = 2 flanges.")

        ell = (ceil(Int, box[1][1]), ceil(Int, box[1][2]))
        u   = (floor(Int, box[2][1]), floor(Int, box[2][2]))
        nx = max(0, u[1] - ell[1])
        ny = max(0, u[2] - ell[2])
        if nx == 0 || ny == 0
            @warn "Box has no interior unit cells"
            return fig, ax
        end

        A = zeros(Int, ny, nx)  # Makie heatmap expects row-major (y,x)
        for iy in 1:ny, ix in 1:nx
            g = [ell[1] + ix - 1, ell[2] + iy - 1]
            cols = active_flats(FG, g)
            rows = active_injectives(FG, g)
            d = (isempty(cols) || isempty(rows)) ? 0 : rankQQ(FG.phi[rows, cols])
            A[iy, ix] = d
        end

        CairoMakie.heatmap!(ax, (ell[1]:u[1]-1), (ell[2]:u[2]-1), A)
        return fig, ax
    end

else
    # No CairoMakie: keep API, but fail at call time (not load time).
    function draw_flange_regions2D(args...; kwargs...)
        _viz2d_missing_dep()
    end
    function draw_constant_subdivision2D(args...; kwargs...)
        _viz2d_missing_dep()
    end
end

end # module
