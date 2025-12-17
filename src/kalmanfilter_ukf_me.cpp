// ------------------------------------------------------------------------------- //
// Advanced Kalman Filtering and Sensor Fusion Course - Unscented Kalman Filter
//
// ####### STUDENT FILE #######
//
// Usage:
// -Rename this file to "kalmanfilter.cpp" if you want to use this code.

#include "kalmanfilter.h"
#include "utils.h"

// -------------------------------------------------- //
// YOU CAN USE AND MODIFY THESE CONSTANTS HERE
constexpr double ACCEL_STD = 0.05;
constexpr double GYRO_STD = 0.01/180.0 * M_PI;
constexpr double INIT_VEL_STD = 12;
constexpr double INIT_PSI_STD = 45.0/180.0 * M_PI;
constexpr double GPS_POS_STD = 3.0;
constexpr double LIDAR_RANGE_STD = 3.0;
constexpr double LIDAR_THETA_STD = 0.02;
// -------------------------------------------------- //

// ----------------------------------------------------------------------- //
// USEFUL HELPER FUNCTIONS
VectorXd normaliseState(VectorXd state)
{
    state(2) = wrapAngle(state(2));
    return state;
}
VectorXd normaliseLidarMeasurement(VectorXd meas)
{
    meas(1) = wrapAngle(meas(1));
    return meas;
}
std::vector<VectorXd> generateSigmaPoints(VectorXd state, MatrixXd cov)
{
    std::vector<VectorXd> sigmaPoints;

    // ----------------------------------------------------------------------- //
    // ENTER YOUR CODE HERE
    int n = state.size();
    double kappa = 3.0 - n;
    double lambda = sqrt(n + kappa);

    MatrixXd sqrtP = cov.llt().matrixL();

    sigmaPoints.push_back(state); // First Sigma Point is the Mean
    for (int i = 0; i < n; ++i)
    {
        sigmaPoints.push_back(state + lambda * sqrtP.col(i));
        sigmaPoints.push_back(state - lambda * sqrtP.col(i));
    }
    // ----------------------------------------------------------------------- //

    return sigmaPoints;
}

std::vector<double> generateSigmaWeights(unsigned int numStates)
{
    std::vector<double> weights;

    // ----------------------------------------------------------------------- //
    // ENTER YOUR CODE HERE
    double n = static_cast<double>(numStates);
    double kappa = 3.0 - n;
    weights.push_back(kappa / (n + kappa)); // Weight for Mean Sigma Point
    for (unsigned int i = 0; i < 2 * numStates; ++i)
    {
        weights.push_back(0.5 / (n + kappa)); // Weights for other Sigma Points
    }
    // ----------------------------------------------------------------------- //

    return weights;
}

VectorXd lidarMeasurementModel(VectorXd aug_state, double beaconX, double beaconY)
{
    VectorXd z_hat = VectorXd::Zero(2);

    // ----------------------------------------------------------------------- //
    // ENTER YOUR CODE HERE
    double x = aug_state(0);
    double y = aug_state(1);
    double psi = aug_state(2);
    double range_noise = aug_state(4);
    double theta_noise = aug_state(5);

    double delta_x = beaconX - x;
    double delta_y = beaconY - y;
    double zhat_range = sqrt(delta_x*delta_x + delta_y*delta_y) + range_noise;
    double zhat_theta = atan2(delta_y,delta_x) - psi + theta_noise;
    z_hat << zhat_range, zhat_theta;

    // ----------------------------------------------------------------------- //

    return z_hat;
}

VectorXd vehicleProcessModel(VectorXd aug_state, double psi_dot, double dt)
{
    VectorXd new_state = VectorXd::Zero(4);

    // ----------------------------------------------------------------------- //
    // ENTER YOUR CODE HERE
    Vector4d state = aug_state.head(4);
    double V = state(3);
    double PSI = state(2);
    new_state(0) = state(0) + V * cos(PSI) * dt;
    new_state(1) = state(1) + V * sin(PSI) * dt;
    new_state(2) = state(2) + (psi_dot + aug_state(4)) * dt;
    new_state(3) = state(3) + aug_state(5) * dt;
    // ----------------------------------------------------------------------- //

    return new_state;
}

void KalmanFilter::handleLidarMeasurement(LidarMeasurement meas, const BeaconMap& map)
{
    if (isInitialised())
    {
        VectorXd state = getState();
        MatrixXd cov = getCovariance();

        // Implement The Kalman Filter Update Step for the Lidar Measurements in the 
        // section below.
        // HINT: Use the normaliseState() and normaliseLidarMeasurement() functions
        // to always keep angle values within correct range.
        // HINT: Do not normalise during sigma point calculation!
        // HINT: You can use the constants: LIDAR_RANGE_STD, LIDAR_THETA_STD
        // HINT: The mapped-matched beacon position can be accessed by the variables
        // map_beacon.x and map_beacon.y
        // ----------------------------------------------------------------------- //
        // ENTER YOUR CODE HERE
        double px = state(0);
        double py = state(1);
        double psi = state(2);
        BeaconData map_beacon = map.getBeaconWithId(meas.id); // Match Beacon with built in Data Association Id
        if (meas.id != -1 && map_beacon.id != -1) // Check that we have a valid beacon match
        {
            // augment state and covariance
            VectorXd aug_state = VectorXd::Zero(6);
            aug_state.head(4) = state;
            MatrixXd aug_cov = MatrixXd::Zero(6,6);
            aug_cov.topLeftCorner(4,4) = cov;
            aug_cov(4,4) = LIDAR_RANGE_STD*LIDAR_RANGE_STD;
            aug_cov(5,5) = LIDAR_THETA_STD*LIDAR_THETA_STD;
            // generate sigma points
            std::vector<VectorXd> sigmaPoints = generateSigmaPoints(aug_state, aug_cov);
            std::vector<double> weights = generateSigmaWeights(aug_state.size());
            // predict measurement of sigma points
            std::vector<VectorXd> predictedMeasSigmaPoints;
            for (const auto& sp : sigmaPoints)
            {
                VectorXd pred_meas_sp = lidarMeasurementModel(sp, map_beacon.x, map_beacon.y);
                predictedMeasSigmaPoints.push_back(pred_meas_sp);
            }

            // predict measurement mean
            VectorXd z_mean = VectorXd::Zero(2);
            for (unsigned int i = 0; i < predictedMeasSigmaPoints.size(); ++i)
            {
                z_mean += weights[i] * predictedMeasSigmaPoints[i];
            }
            z_mean = normaliseLidarMeasurement(z_mean);
            // predict measurement covariance S
            Matrix2d S = Matrix2d::Zero();
            for (unsigned int i = 0; i < predictedMeasSigmaPoints.size(); ++i)
            {
                VectorXd diff = predictedMeasSigmaPoints[i] - z_mean;
                S += weights[i] * (diff * diff.transpose());    
            }
            // calculate cross correlation matrix
            MatrixXd Pxz = MatrixXd::Zero(4,2);
            for (unsigned int i = 0; i < predictedMeasSigmaPoints.size(); ++i)
            {
                VectorXd x_diff = sigmaPoints[i].head(4) - state;
                VectorXd z_diff = predictedMeasSigmaPoints[i] - z_mean;
                Pxz += weights[i] * (x_diff * z_diff.transpose());
            }

            // calculate Kalman gain K
            MatrixXd K = Pxz * S.inverse();
            // update state and covariances
            state += K * (Vector2d(meas.range, meas.theta) - z_mean);
            state = normaliseState(state);
            // update covariance
            cov -= K * S * K.transpose();
        }
        // ----------------------------------------------------------------------- //

        setState(state);
        setCovariance(cov);
    }
}

void KalmanFilter::predictionStep(GyroMeasurement gyro, double dt)
{
    if (isInitialised())
    {
        VectorXd state = getState();
        MatrixXd cov = getCovariance();

        // Implement The Kalman Filter Prediction Step for the system in the  
        // section below.
        // HINT: Assume the state vector has the form [PX, PY, PSI, V].
        // HINT: Use the Gyroscope measurement as an input into the prediction step.
        // HINT: You can use the constants: ACCEL_STD, GYRO_STD
        // HINT: Use the normaliseState() function to always keep angle values within correct range.
        // HINT: Do NOT normalise during sigma point calculation!
        // ----------------------------------------------------------------------- //
        // ENTER YOUR CODE HERE
        VectorXd aug_state = VectorXd::Zero(6);
        aug_state.head(4) = state;
        MatrixXd aug_cov = MatrixXd::Zero(6,6);
        aug_cov.topLeftCorner(4,4) = cov;
        aug_cov(4,4) = GYRO_STD*GYRO_STD;
        aug_cov(5,5) = ACCEL_STD*ACCEL_STD;

        std::vector<VectorXd> sigmaPoints = generateSigmaPoints(aug_state, aug_cov);
        std::vector<double> weights = generateSigmaWeights(aug_state.size());
        std::vector<VectorXd> predictedSigmaPoints;
        for (const auto& sp : sigmaPoints)
        {
            VectorXd pred_sp = vehicleProcessModel(sp, gyro.psi_dot, dt);
            predictedSigmaPoints.push_back(pred_sp);
        }
        // Predict State Mean
        VectorXd pred_state = VectorXd::Zero(4);
        for (unsigned int i = 0; i < predictedSigmaPoints.size(); ++i)
        {
            pred_state += weights[i] * predictedSigmaPoints[i];
        }
        pred_state = normaliseState(pred_state);
        // Predict State Covariance
        MatrixXd pred_cov = MatrixXd::Zero(4,4);
        for (unsigned int i = 0; i < predictedSigmaPoints.size(); ++i)
        {
            VectorXd diff = predictedSigmaPoints[i] - pred_state;
            diff(2) = wrapAngle(diff(2)); // Only wrap the angle difference
            pred_cov += weights[i] * (diff * diff.transpose());
        }
        // ----------------------------------------------------------------------- //

        setState(pred_state);
        setCovariance(pred_cov);
    } 
}

void KalmanFilter::handleGPSMeasurement(GPSMeasurement meas)
{
    // All this code is the same as the LKF as the measurement model is linear
    // so the UKF update state would just produce the same result.
    if(isInitialised())
    {
        VectorXd state = getState();
        MatrixXd cov = getCovariance();

        VectorXd z = Vector2d::Zero();
        MatrixXd H = MatrixXd(2,4);
        MatrixXd R = Matrix2d::Zero();

        z << meas.x,meas.y;
        H << 1,0,0,0,0,1,0,0;
        R(0,0) = GPS_POS_STD*GPS_POS_STD;
        R(1,1) = GPS_POS_STD*GPS_POS_STD;

        VectorXd z_hat = H * state;
        VectorXd y = z - z_hat;
        MatrixXd S = H * cov * H.transpose() + R;
        MatrixXd K = cov*H.transpose()*S.inverse();

        state = state + K*y;
        cov = (MatrixXd::Identity(4,4) - K*H) * cov;

        setState(state);
        setCovariance(cov);
    }
    else
    {
        // You may modify this initialisation routine if you can think of a more
        // robust and accuracy way of initialising the filter.
        // ----------------------------------------------------------------------- //
        // YOU ARE FREE TO MODIFY THE FOLLOWING CODE HERE

        VectorXd state = Vector4d::Zero();
        MatrixXd cov = Matrix4d::Zero();

        state(0) = meas.x;
        state(1) = meas.y;
        cov(0,0) = GPS_POS_STD*GPS_POS_STD;
        cov(1,1) = GPS_POS_STD*GPS_POS_STD;
        cov(2,2) = INIT_PSI_STD*INIT_PSI_STD;
        cov(3,3) = INIT_VEL_STD*INIT_VEL_STD;

        setState(state);
        setCovariance(cov);

        // ----------------------------------------------------------------------- //
    }             
}

void KalmanFilter::handleLidarMeasurements(const std::vector<LidarMeasurement>& dataset, const BeaconMap& map)
{
    // Assume No Correlation between the Measurements and Update Sequentially
    for(const auto& meas : dataset) {handleLidarMeasurement(meas, map);}
}

Matrix2d KalmanFilter::getVehicleStatePositionCovariance()
{
    Matrix2d pos_cov = Matrix2d::Zero();
    MatrixXd cov = getCovariance();
    if (isInitialised() && cov.size() != 0){pos_cov << cov(0,0), cov(0,1), cov(1,0), cov(1,1);}
    return pos_cov;
}

VehicleState KalmanFilter::getVehicleState()
{
    if (isInitialised())
    {
        VectorXd state = getState(); // STATE VECTOR [X,Y,PSI,V,...]
        return VehicleState(state[0],state[1],state[2],state[3]);
    }
    return VehicleState();
}

void KalmanFilter::predictionStep(double dt){}