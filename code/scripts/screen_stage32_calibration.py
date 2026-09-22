"""One bounded expanded-neighborhood calibration screen after Stage31."""
import screen_stage31_calibration as screen

# Include the unchanged Stage30 reference plus the positive neighborhood selected
# by Stage31. This is the final scalar grid; do not adapt it again after results.
screen.TEMPERATURES = (1., 1.025, 1.05, 1.075, 1.10, 1.125)
screen.PRIOR_WEIGHTS = (0., .025, .05, .075, .10, .125)
screen.GATE_SHIFTS = (0., .125, .25, .375, .50, .625)
screen.MIXTURE_WEIGHTS = (.10, .1125, .125, .1375, .15)


if __name__ == "__main__":
    screen.main()
