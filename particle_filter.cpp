/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

/*
	Created by Jonas Chan for Udacity CARND Kidnapped Vehicle Project
*/
#define _USE_MATH_DEFINES
#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;



void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	default_random_engine gen;

	// Set the number of particles
	num_particles = 101;

	// Create a normal Gaussian distribution for x, y and theta with mean = 0 and 
	normal_distribution<double> dist_x(0, std[0]);
	normal_distribution<double> dist_y(0, std[1]);
	normal_distribution<double> dist_theta(0, std[2]);

	// Initialise all the particles to first position based on the x, y, theta and their uncertainties from the GPS as well as
	// initialise all the particle weights to 1.
	for (int i = 0; i < num_particles; i++)
	{
		Particle particle;
		particle.id = i;
		particle.x = x;
		particle.y = y;
		particle.theta = theta;
		particle.weight = 1.0;

		// Add noise to each particle based on the created normal Gaussian distribution
		particle.x += dist_x(gen);
		particle.y += dist_y(gen);
		particle.theta += dist_theta(gen);

		// Add the newly generated particle to the list of particles
		particles.push_back(particle);
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	default_random_engine gen;

	// define normal distributions for sensor noise
	normal_distribution<double> noise_x(0, std_pos[0]);
	normal_distribution<double> noise_y(0, std_pos[1]);
	normal_distribution<double> noise_theta(0, std_pos[2]);

	for (int i = 0; i < num_particles; i++) {

		// calculate new state
		if (fabs(yaw_rate) < 0.00001) 
		{
			particles[i].x += velocity * delta_t * cos(particles[i].theta) + noise_x(gen);
			particles[i].y += velocity * delta_t * sin(particles[i].theta) + noise_y(gen);
		}
		else 
		{
			particles[i].x += (velocity / yaw_rate * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta))) + noise_x(gen);
			particles[i].y += (velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t))) + noise_y(gen);
			particles[i].theta += (yaw_rate * delta_t) + noise_theta(gen);
		}
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations)
{
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.


	// Nearest neighbour technique - Take the closest measurement as the correct correspondant
	for (unsigned int i = 0; i < observations.size(); i++)
	{
		// Get the current observed landmark
		LandmarkObs observedLandmark = observations[i];

		// Initialise the minimum distance to the maximum possible distance
		double minimumDistance = numeric_limits<double>::max();

		// Initialise the id of the landmark from the placeholder map associated with the observation
		int map_id = -1;

		// Run through all the predictions and select the predicted landmark closest to the observed landmark
		for (unsigned int a = 0; a < predicted.size(); a++)
		{
			// Get the current predicted landmark
			LandmarkObs predictedLandmark = predicted[a];

			// Calculate the distance between the observed and predicted landmark using the dist() function in
			// the helper_functions.h
			double calculatedDistance = dist(observedLandmark.x, observedLandmark.y, predictedLandmark.x, predictedLandmark.y);

			if (calculatedDistance < minimumDistance)
			{
				// Now that we've found the distance closest to the landmark, set this as the minimum distance
				// so it can be compared again for the next prediction
				minimumDistance = calculatedDistance;

				// Get the id of the predicted landmark
				map_id = predictedLandmark.id;
			}
		}

		// Now that we've found the predicted landmark closest to the observed landmark,
		// we set the predicted landmark id to the observed landmark id.
		observations[i].id = map_id;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
	std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account 
	//   for the fact that the map's y-axis actually points downwards.)
	//   http://planning.cs.uiuc.edu/node99.html

	// ***Quoting lectures here*** 
	// The goal of the update step is to obtain the weight of each particle. One way to update the weight of the
	// particle is to use the multivariate gaussian probability density function for each measurement and combine the 
	// likelihood of all the measurement by taking thier product.

	// Updating for each particle
	for (int i = 0; i < num_particles; i++)
	{
		// Obtain the particle's x and y coordinates as well as the heading of the particle
		double particle_x = particles[i].x;
		double particle_y = particles[i].y;
		double particle_theta = particles[i].theta;

		// Get predicted landmark coordinates

		// Create a vector to store the landmarks coordinates
		vector<LandmarkObs> landmark;

		for (unsigned int i = 0; i < map_landmarks.landmark_list.size(); i++)
		{
			float landmark_x = map_landmarks.landmark_list[i].x_f;
			float landmark_y = map_landmarks.landmark_list[i].y_f;
			int landmark_id = map_landmarks.landmark_list[i].id_i;

			// Calculate the distance between the particle and the landmark
			double calculated_distance_x = landmark_x - particle_x;
			double calculated_distance_y = landmark_y - particle_y;
			double minimum_distance = sensor_range;

			// Here we only consider landmarks which are within the range of the sensor.
			// If it is, we append its coordinates to the landmark variable 
			if (fabs(calculated_distance_x) <= minimum_distance)
			{
				if (fabs(calculated_distance_y) <= minimum_distance)
				{
					landmark.push_back(LandmarkObs{ landmark_id, landmark_x, landmark_y });
				}
			}
		}

		// Create a vector to store the transformed observations
		vector<LandmarkObs> transformed_observations;

		// Predict measurements to all map landmarks.
		// 1.	Get predicted landmarks in local coordinates
		//		- Transform vehicle's observation to local coordinates using the following trigonometric equations
		//			transformed_x = x * cos(theta) - y * sin(theta) + translation_x
		//			transformed_y = x * sin(theta) + y * cos(theta) + translation_y
					
		for (unsigned int i = 0; i < observations.size(); i++)
		{
			double transformed_observation_x = observations[i].x * cos(particle_theta) - observations[i].y * sin(particle_theta) + particle_x;
			double transformed_observation_y = observations[i].x * sin(particle_theta) + observations[i].y * cos(particle_theta) + particle_y;
			
			// Append the newly transformed observations into the transformed_observation variable
			transformed_observations.push_back(LandmarkObs{ observations[i].id, transformed_observation_x, transformed_observation_y});
		}

		

		// Use the dataAssosiation method for predicted and observed landmarks
		dataAssociation(landmark, transformed_observations);

		// Initialise the unnormalised weight for the particle
		particles[i].weight = 1.0;

		// Calculate the new weight of each particle using the multivariate Gaussian probability density function
		for (unsigned int a = 0; a < transformed_observations.size(); a++)
		{
			// Create variables to hold the predicted landmark and associated landmark
			double associated_landmark_x = transformed_observations[a].x;
			double associated_landmark_y = transformed_observations[a].y;
			int associated_landmark_id = transformed_observations[a].id;

			double predicted_landmark_x;
			double predicted_landmark_y;

			for (unsigned int j = 0; j < landmark.size(); j++)
			{
				if (landmark[j].id == associated_landmark_id)
				{
					predicted_landmark_x = landmark[j].x;
					predicted_landmark_y = landmark[j].y;
				}
			}

			// Define some math calculations for the multivariate Gaussian probability density function
			double x_diff = predicted_landmark_x - associated_landmark_x;
			double y_diff = predicted_landmark_y - associated_landmark_y;
			double variance_x = pow(std_landmark[0], 2);
			double variance_y = pow(std_landmark[1], 2);
			double covariance_xy = std_landmark[0] * std_landmark[1];
			
			// Calculate weight using the multivariate Gaussian probability density function
			double weight = (1 / (2 * M_PI * covariance_xy)) * exp(-(pow(x_diff, 2) / (2 * variance_x) + (pow(y_diff, 2) / (2 * variance_y))));
			particles[i].weight *= weight;
		}		
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	default_random_engine gen;

	// Get the updated current weights
	vector<double> updatedWeights;

	for (int i = 0; i < num_particles; i++)
	{
		double weight = particles[i].weight;
		updatedWeights.push_back(weight);
	}

	uniform_int_distribution<int> uniintdist(0, num_particles - 1);
	auto uniform_index = uniintdist(gen);

	// Get maximum weight using the *max_element which returns values
	// *Note to self: max_element returns iterators not values!!!.
	double maximum_weight = *max_element(updatedWeights.begin(), updatedWeights.end());

	uniform_real_distribution<double> unirealdist(0.0, maximum_weight);
	double resamplingBeta = 0.0;

	vector<Particle> new_particles;
	for (int i = 0; i < num_particles; i++) 
	{
		resamplingBeta += unirealdist(gen) * 2.0;
		
		while (resamplingBeta > updatedWeights[uniform_index])
		{
			resamplingBeta -= updatedWeights[uniform_index];
			uniform_index = (uniform_index + 1) % num_particles;
		}

		new_particles.push_back(particles[uniform_index]);
	}

	particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
