# Robotarium SLAM Demo

## Description

This directory contains a Robotarium Demo/Experiment demonstrating real-time SLAM.  This experiment divides the robots into two teams.  The first team is the landmarkers.  The landmarker robots are responsible for staying put at different locations in the robotarium to add features for the searching robots to SLAM off of.  The second team is the seekers who wander around the environment using their distance sensors to detect the landmarking robots.  Right now, the experiment works best with a single seeker because the bearing range factors added by the robots tend to be noisy.  From what I can tell, these factors are made even more noisy by the fact that the robots have finite width and I'm attempting to correct for this to find their centers by subtracting half a robot radius.  This experiment can work with multiple seekers (and is supported), but it tends to be a lot less consistent as the covariance on robot poses tends to expand by quite a bit meaning that label correspondence can get confused.

## Running

This project is setup to use `uv` and `just`.  The justfile contains a set of useful commands for interacting with the project, but the main jist is that running `just run` runs `uv run python main.py` and `just test` runs `uv run python -m unittest discover -s tests` to run the tests I wrote.
