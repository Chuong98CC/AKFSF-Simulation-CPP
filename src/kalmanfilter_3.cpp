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
constexpr double ACCEL_STD = 1.0;
constexpr double GYRO_STD = 0.01/180.0 * M_PI;
constexpr double INIT_VEL_STD = 10.0;
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
    double V = aug_state(3);
    double gyro_bias = aug_state(4);
    double range_noise = aug_state(5);
    double theta_noise = aug_state(6);

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
    VectorXd new_state = VectorXd::Zero(5); // [PX, PY, PSI, V, GYRO_BIAS]

    // ----------------------------------------------------------------------- //
    // ENTER YOUR CODE HERE
    VectorXd state = aug_state.head(5);
    double psi = state(2);
    double V = state(3);
    double gyro_bias = state(4);
    new_state(0) = state(0) + V * cos(psi) * dt;        
    new_state(1) = state(1) + V * sin(psi) * dt;
    new_state(2) = state(2) + (psi_dot - gyro_bias + aug_state(5)) * dt;
    new_state(3) = state(3) + aug_state(6) * dt;
    new_state(4) = state(4);
    // ----------------------------------------------------------------------- //

    return new_state;
}

void KalmanFilter::handleLidarMeasurement(LidarMeasurement meas, const BeaconMap& map)
{
    if (isInitialised()) {
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
            VectorXd aug_state = VectorXd::Zero(7);
            aug_state.head(5) = state;
            MatrixXd aug_cov = MatrixXd::Zero(7,7);
            aug_cov.topLeftCorner(5,5) = cov;
            aug_cov(5,5) = LIDAR_RANGE_STD*LIDAR_RANGE_STD;
            aug_cov(6,6) = LIDAR_THETA_STD*LIDAR_THETA_STD;
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
            // z_mean = normaliseLidarMeasurement(z_mean);
            // predict measurement covariance S
            Matrix2d S = Matrix2d::Zero();
            for (unsigned int i = 0; i < predictedMeasSigmaPoints.size(); ++i)
            {
                VectorXd diff = normaliseLidarMeasurement(predictedMeasSigmaPoints[i] - z_mean);
                S += weights[i] * (diff * diff.transpose());    
            }
            // calculate cross correlation matrix
            MatrixXd Pxz = MatrixXd::Zero(5,2);
            for (unsigned int i = 0; i < predictedMeasSigmaPoints.size(); ++i)
            {
                VectorXd x_diff = normaliseState(sigmaPoints[i].head(5) - state);
                VectorXd z_diff = normaliseLidarMeasurement(predictedMeasSigmaPoints[i] - z_mean);
                Pxz += weights[i] * (x_diff * z_diff.transpose());
            }

            // calculate Kalman gain K
            MatrixXd K = Pxz * S.inverse();
            // update state and covariances
            state += K * normaliseLidarMeasurement(Vector2d(meas.range, meas.theta) - z_mean);
            cov -= K * S * K.transpose();
            // state = normaliseState(state);
            // update covariance
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
        VectorXd aug_state = VectorXd::Zero(7); // [PX, PY, PSI, V, GYRO_BIAS, GYRO_NOISE, ACCEL_NOISE] 
        aug_state.head(5) = state;
        MatrixXd aug_cov = MatrixXd::Zero(7,7);
        aug_cov.topLeftCorner(5,5) = cov;
        aug_cov(5,5) = GYRO_STD*GYRO_STD;
        aug_cov(6,6) = ACCEL_STD*ACCEL_STD;

        std::vector<VectorXd> sigmaPoints = generateSigmaPoints(aug_state, aug_cov);
        std::vector<double> weights = generateSigmaWeights(aug_state.size());
        std::vector<VectorXd> predictedSigmaPoints;
        for (const auto& sp : sigmaPoints)
        {
            VectorXd pred_sp = vehicleProcessModel(sp, gyro.psi_dot, dt);
            predictedSigmaPoints.push_back(pred_sp);
        }
        // Predict State Mean
        VectorXd pred_state = VectorXd::Zero(5);
        for (unsigned int i = 0; i < predictedSigmaPoints.size(); ++i)
        {
            pred_state += weights[i] * predictedSigmaPoints[i];
        }
        pred_state = normaliseState(pred_state);
        // Predict State Covariance
        MatrixXd pred_cov = MatrixXd::Zero(5,5);
        for (unsigned int i = 0; i < predictedSigmaPoints.size(); ++i)
        {
            VectorXd diff = normaliseState(predictedSigmaPoints[i] - pred_state);
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
    if(init_GPS)
    {
        VectorXd state = getState();
        MatrixXd cov = getCovariance();

        VectorXd z = Vector2d::Zero();
        MatrixXd H = MatrixXd(2,5);
        MatrixXd R = Matrix2d::Zero();

        z << meas.x,meas.y;
        H(0,0) = 1;
        H(1,1) = 1;
        R(0,0) = GPS_POS_STD*GPS_POS_STD;
        R(1,1) = GPS_POS_STD*GPS_POS_STD;

        VectorXd z_hat = H * state;
        VectorXd y = z - z_hat;
        MatrixXd S = H * cov * H.transpose() + R;
        MatrixXd K = cov*H.transpose()*S.inverse();

        state = state + K*y;
        cov = (MatrixXd::Identity(5,5) - K*H) * cov;

        setState(state);
        setCovariance(cov);
    }
    else
    {
        // You may modify this initialisation routine if you can think of a more
        // robust and accuracy way of initialising the filter.
        // ----------------------------------------------------------------------- //
        // YOU ARE FREE TO MODIFY THE FOLLOWING CODE HERE

        VectorXd state = VectorXd::Zero(5);
        MatrixXd cov = MatrixXd::Zero(5,5);

        state(0) = meas.x;
        state(1) = meas.y;
        cov(0,0) = GPS_POS_STD*GPS_POS_STD;
        cov(1,1) = GPS_POS_STD*GPS_POS_STD;
        cov(2,2) = INIT_PSI_STD*INIT_PSI_STD;
        cov(3,3) = INIT_VEL_STD*INIT_VEL_STD;

        setState(state);
        setCovariance(cov);
        init_GPS = true;  // Mark GPS as initialized
        // m_initialised = init_GPS && init_lidar;
        // ----------------------------------------------------------------------- //
    }             
}

void KalmanFilter::handleLidarMeasurements(const std::vector<LidarMeasurement>& dataset, const BeaconMap& map)
{
    auto meas = dataset[0];
    if (meas.id != -1) {  
        // for simulation that provides data association id directly
        for(const auto& meas : dataset) {
            handleLidarMeasurement(meas, map);
        }
        return;
    }
    // no data association id provided, need to perform data association
    if (!init_GPS){
        // cannot process lidar measurements before GPS initialisation
        return;
    }
    // add data association here
    std::vector<LidarMeasurement> dataset_copy = dataset;
    VectorXd state = getState();
    MatrixXd cov = getCovariance();
    double px = state(0);
    double py = state(1);
    double psi = state(2);

    // find the maximum lidar range to search for associated beacons
    double max_lidar_range = std::max_element(dataset.begin(), dataset.end(),
        [](const auto& a, const auto& b) { return a.range < b.range; })->range;
    // get list of beacons within lidar range
    double range_margin = 3.0 * LIDAR_RANGE_STD + 6*sqrt(cov(0,0) + cov(1,1));
    std::vector<BeaconData> nearby_beacons = map.getBeaconsWithinRange(px, py, max_lidar_range+ range_margin); // add some margin
    int num_beacons = nearby_beacons.size();
    int num_measurements = dataset.size();
    if (num_beacons<1 || num_measurements<1){
        return;
    }

    // build association matrix
    Eigen::MatrixXd association_matrix = Eigen::MatrixXd::Zero(num_measurements, num_beacons);
    for (int i = 0; i < num_measurements; ++i) {
        double meas_range = dataset[i].range;
        double meas_bearing = dataset[i].theta;
        for (int j = 0; j < num_beacons; ++j) {
            double dx = nearby_beacons[j].x - px;
            double dy = nearby_beacons[j].y - py;
            double expected_range = sqrt(dx*dx + dy*dy);
            double range_diff = fabs(meas_range - expected_range)/ LIDAR_RANGE_STD;
            // if (init_lidar){
            //     // if filter is initialised, use both range and bearing for association
            //     double expected_bearing = wrapAngle(atan2(dy, dx) - psi);
            //     double bearing_diff = fabs(meas_bearing - expected_bearing)/ LIDAR_THETA_STD;
            //     association_matrix(i, j) = sqrt(range_diff*range_diff + bearing_diff*bearing_diff);          
            // }
            // else{
            //     // if filter is not initialised, use only range for association. we need to estimate the heading
            //     association_matrix(i, j) = range_diff;
            // }
            association_matrix(i, j) = range_diff;
        }
    }
    // solve assignment problem: 
    // 1 lidar measurement can be associated to only 1 beacon, 
    // but 1 beacon can be associated to multiple lidar measurements
    for (int i = 0; i < num_measurements; ++i) {
        Eigen::Index minIndex;
        double minValue = association_matrix.row(i).minCoeff(&minIndex);
        // dataset_copy[i].id = nearby_beacons[minIndex].id;    
        if (minValue < 3.0) { // threshold for association based on 3-sigma rule
            dataset_copy[i].id = nearby_beacons[minIndex].id;    
        }
    }
    // find the bearing if not initialised
    // if (!isInitialised()){
    if (!init_lidar){
        std::vector<double> psi_candidates;
        for (const auto& meas : dataset_copy) {
            if (meas.id != -1){
                BeaconData assoc_beacon = map.getBeaconWithId(meas.id);
                double dx = assoc_beacon.x - px;
                double dy = assoc_beacon.y - py;
                double expected_bearing = atan2(dy, dx);
                psi = wrapAngle(expected_bearing - meas.theta);
                psi_candidates.push_back(psi);
            }
        }
        if (!psi_candidates.empty()) {
            // average the candidate headings through circular mean to handle angle wrapping
            double sum_sin = 0.0;
            double sum_cos = 0.0;
            for (double angle : psi_candidates) {
                sum_sin += sin(angle);
                sum_cos += cos(angle);
            }
            psi = atan2(sum_sin, sum_cos);
            state(2) = psi;
            setState(state);
            init_lidar = true;  // Mark lidar as initialized (heading estimated)
            // m_initialised = init_GPS && init_lidar;
        }
    }
    

    for(const auto& meas : dataset_copy) {
        // add data association here
        handleLidarMeasurement(meas, map);
    }
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