%% Processing recordings
%IPM 165
clear all;
close all;

wavFile_CW_All = {'165_car_rangetowards_1m_1.wav';
                  '165_car_rangewaway_1m_1.wav';
                  '365_carf.wav';
                  '165_Bakkie_slowingdown_toward.wav';
                  '165_Bakkie_speedingup_towards.wav';
                  '165_car_60kmph.wav';
                  '165_Car_away.wav';
                  '165_car_towards.wav';
                  '165_car_speedingup.wav'};
              
RecordingNo2Process = 1;             

wavFile = wavFile_CW_All{RecordingNo2Process};

% Input parameters
CPI = 0.5; % seconds
PFA = 10^-5;
RefWindow = 30; 
GaurdCells = 2;
overlapPercent = 60;            

% Constants
c = 299e6; % (m/s) speed of light
fc = 24e9; % (Hz) Center frequency 
maxSpeed_km_hr = 60; % (km/hr) maximum speed to display
maxSpeed_m_s = 5;

% computations
lamda = c/fc;

% % read the raw wave data
[Y,fs] = audioread(wavFile,'native');
% if size(Y, 2) < 2
%     error('Audio file must have at least two channels.');
% end
%y = -Y(:,2); % Received signal at baseband

% Determine the number of channels
[numSamples, numChannels] = size(Y);

if numChannels < 2
    % Handle mono file
    disp('Mono file detected. Using the single available channel.');
    y = -Y(:,1);  % Use the single channel
else
    % Handle stereo file
    y = -Y(:,2);  % Use the second channel
end

% Compute the spectrogram 
windowLength = round(CPI * fs); % Define the window length
nfft = 2 ^ nextpow2(windowLength); % Ensure it's a power of 2

% Call the JakeSpectrogram function
[fX, tX, specMatrix] = JakeSpectrogram(y, fs, windowLength, overlapPercent, nfft);


speed_m_per_sec = fX*lamda/2; % Calculate the speed on the object 
% speed_m_per_sec_Idx = find((speed_m_per_sec <= maxSpeed_m_s) & (speed_m_per_sec >= 0));
% SpeedVectorOfInterest = speed_m_per_sec(speed_m_per_sec_Idx);
% S_OfInterest = specMatrix(speed_m_per_sec_Idx, :);

speed_km_per_hr = speed_m_per_sec*(60*60/1000);
speed_km_per_hr_Idx = find((speed_km_per_hr <= maxSpeed_km_hr) & (speed_km_per_hr >= 5));
SpeedVectorOfInterest = speed_km_per_hr(speed_km_per_hr_Idx);
S_OfInterest = specMatrix(speed_km_per_hr_Idx, :);

% select version of S_OfInterestToPlot

% This version normalizes the matrix column by column with a detection threshold

% % Define a detection threshold
% detectionThreshold = max(abs(S_OfInterest(:,1))); 
% 
% [K, L] = size(S_OfInterest);  % K corresponds to speed bins (rows), L to time bins (columns)
% S_OfInterestToPlot = zeros(K, L);  % Preallocate the output matrix
% 
% % Estimate the noise level from the first column
% noise_level_estimate = mean(abs(S_OfInterest(:,1)));
% 
% % Calculate average intensity per row
% row_intensity = mean(abs(S_OfInterest), 2);
% 
% % Calculate median row intensity as a reference
% median_row_intensity = median(row_intensity);
% 
% % Set a threshold factor for detecting high-intensity rows
% intensity_threshold_factor = 4; % You can adjust this factor based on your data
% high_intensity_threshold = intensity_threshold_factor * median_row_intensity;
% 
% % Loop through time bins and process the data
% for i = 1:L
%    S_OfInterest_mod(:,i) = abs(S_OfInterest(:,i)) - 2*noise_level_estimate;
% 
%    % Ensure non-negative values after noise subtraction
%    S_OfInterest_mod(:, i) = max(S_OfInterest_mod(:, i), 0);
% 
%    % Check for high-intensity rows (horizontal lines) and normalise them
%    for k = 1:K
%        if row_intensity(k) > high_intensity_threshold
%            % Normalise the entire row to reduce visibility
%            S_OfInterest_mod(k, i) = S_OfInterest_mod(k, i) / row_intensity(k);
%        end
%    end
% 
%    maxVal = max(max(abs(S_OfInterest_mod(:, i))));
% 
%    if maxVal > detectionThreshold
%        % If the max value in the column is above the threshold, normalise
%        S_OfInterestToPlot(:, i) = abs(S_OfInterest_mod(:, i)) / maxVal;
%    else
%        % If below threshold, set column values to lower intensity
%        S_OfInterestToPlot(:, i) = abs(S_OfInterest_mod(:,i)) / detectionThreshold;  
%    end
% end

% Plot the spectrogram using your desired plotting method



% % % Define a detection threshold 
detectionThreshold = 1*max(abs(S_OfInterest(:,1))); 

[K, L] = size(S_OfInterest);  % K corresponds to speed bins (rows), L to time bins (columns)
S_OfInterestToPlot = zeros(K, L);  % Preallocate the output matrix
noise_level_estimate = mean(abs(S_OfInterest(:,1)));
for i = 1:L
   S_OfInterest_mod(:,i) = abs(S_OfInterest(:,i)) - 2*noise_level_estimate;
   % Ensure non-negative values after noise subtraction
   S_OfInterest_mod(:, i) = max(S_OfInterest_mod(:, i), 0);

   maxVal = max(max(abs(S_OfInterest_mod(:, i))));

   if maxVal > detectionThreshold
       % If the max value in the column is above the threshold, normalise
       %disp(maxVal);
       S_OfInterestToPlot(:, i) = abs(S_OfInterest_mod(:, i)) / maxVal;
   else
       % If below threshold, set column values to zero or some lower intensity
       S_OfInterestToPlot(:, i) = abs(S_OfInterest_mod(:,i))/(5*detectionThreshold);  
   end
end

%OR

%This version normalizes the matrix collumn by collumn 
% [K, L] = size(S_OfInterest);  % K corresponds to speed bins (rows), L to time bins (columns)
% S_OfInterestToPlot = zeros(K, L);  % Preallocate the output matrix
% for i = 1:L
%    S_OfInterestToPlot(:, i) = abs(S_OfInterest(:, i)) / max(max(abs(S_OfInterest(:, i))));
% end

% OR 
%This version normalizes the matrix based on the largest value in the
%entire matrix 
S_OfInterestToPlot = 1*abs(S_OfInterest) / (1*max(max(abs(S_OfInterest))));

% Plot the spectrogram 
clims = [-40 0];
figure; 
imagesc(tX,SpeedVectorOfInterest,20*log10(S_OfInterestToPlot), clims);
xlabel('Time (s)');
ylabel('Speed (km/h)');
grid on;
colorbar;
colormap('jet');
axis xy;

% Perform CA-CFAR detection
[Detect] = CA_CFAR(S_OfInterest, PFA, RefWindow, GaurdCells);

% Find indices of detections
[y_idx, x_idx] = find(Detect == 1);

% Extract the detected speeds
detected_speeds = SpeedVectorOfInterest(y_idx);

% Calculate the mean and standard deviation of the detected speeds
mean_speed = mean(detected_speeds);
std_speed = std(detected_speeds);

% Define how many standard deviations to keep detections 
n_std_dev = 2;

% Filter detections that are within n standard deviations of the mean
valid_detections = abs(detected_speeds - mean_speed) <= n_std_dev * std_speed;

% Update the indices of valid detections
y_idx_filtered = y_idx(valid_detections);
x_idx_filtered = x_idx(valid_detections);

% % Plot the spectrogram with detection markers
figure; 
imagesc(tX, SpeedVectorOfInterest, 20 * log10(abs(S_OfInterestToPlot)), clims);
hold on;
[y_idx, x_idx] = find(Detect == 1);

% Convert indices to coordinates for plotting
plot(tX(x_idx), SpeedVectorOfInterest(y_idx), 'wx', 'markersize', 10, 'DisplayName', 'Detections');
xlabel('Time (s)');
ylabel('Speed (km/h)');
grid on;

% Create the legend and set background and text colors
hLegend = legend('Detections');
set(hLegend, 'TextColor', 'white', 'Color', 'black');  % Change text to white and background to black

colorbar;
colormap('jet');
axis xy;
hold off;

% Calculate the mean speed of the filtered detections
filtered_speed = mean(SpeedVectorOfInterest(y_idx_filtered));
fprintf('The average filtered speed is: %.2f km/h\n', filtered_speed);





