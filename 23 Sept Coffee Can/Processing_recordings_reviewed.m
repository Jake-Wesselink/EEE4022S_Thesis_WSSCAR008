%% Processing recordings
% Coffee Can
clear all;
close all;

wavFile_CW_All = {'MIT_jammie_away_2.wav';
                  'MIT_jammie_towards.wav';
                  'MIT_busy_Roar_2.wav';
                  'busy_road_1_MIT.wav';
                  'MIT_jammie_towards_and_away.wav'};
              
RecordingNo2Process = 2;             

wavFile = wavFile_CW_All{RecordingNo2Process};

% Input parameters
CPI = 0.5; % seconds
PFA = 10^-7;
RefWindow = 10; 
GaurdCells = 2;
overlapPercent = 60;            

% Constants
c = 299e6; % (m/s) speed of light
fc = 2.4e9; % (Hz) Center frequency 
maxSpeed_km_hr = 80; % (km/hr) maximum speed to display
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

%This value accounts for noise and low intensity columns 

% Define a detection threshold 
detectionThreshold = 10*max(abs(S_OfInterest(:,1))); 
% This version normalizes the matrix column by column with a detection threshold
[K, L] = size(S_OfInterest);  
S_OfInterestToPlot = zeros(K, L);  
noise_level_estimate = mean(abs(S_OfInterest(:,1)));
for i = 1:L

   S_OfInterest_mod(:,i) = abs(S_OfInterest(:,i)) - noise_level_estimate;
   % Ensure non-negative values after noise subtraction
   S_OfInterest_mod(:, i) = max(S_OfInterest_mod(:, i), 0);

   maxVal = max(max(abs(S_OfInterest_mod(:, i))));

   if maxVal > detectionThreshold
       % If the max value in the column is above the threshold, normalise
       % to that value
       S_OfInterestToPlot(:, i) = abs(S_OfInterest_mod(:, i)) / maxVal;
   else
       % If below threshold, normalise column values to the detection
       % threshold
       S_OfInterestToPlot(:, i) = abs(S_OfInterest_mod(:,i))/(detectionThreshold); 
   end
end

% OR 

%This version normalizes the matrix collumn by collumn 
[K, L] = size(S_OfInterest);  % K corresponds to speed bins (rows), L to time bins (columns)
S_OfInterestToPlot = zeros(K, L);  % Preallocate the output matrix
for i = 1:L
   S_OfInterestToPlot(:, i) = abs(S_OfInterest(:, i)) / max(max(abs(S_OfInterest(:, i))));
end

% OR

%This version normalizes the matrix based on the largest value in the
%entire matrix 
%S_OfInterestToPlot = abs(S_OfInterest) / max(max(abs(S_OfInterest)));

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


% % Plot the spectrogram with detection markers
figure; 
imagesc(tX, SpeedVectorOfInterest, 20 * log10(S_OfInterestToPlot), clims);
hold on;
[y_idx, x_idx] = find(Detect == 1);

% Convert indices to coordinates for plotting
plot(tX(x_idx), SpeedVectorOfInterest(y_idx), 'kx', 'markersize', 10, 'DisplayName', 'Detections');
xlabel('Time (s)');
ylabel('Speed (km/h)');
grid on;
legend('Detections')
colorbar;
colormap('jet');
axis xy;
hold off;
% 
% speed = mean(SpeedVectorOfInterest(y_idx));
% fprintf('The average speed is: %.2f km/h\n', speed);



