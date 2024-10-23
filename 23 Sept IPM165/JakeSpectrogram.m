function [fX, tX, specMatrix] = JakeSpectrogram(signal, fs, windowLength, overlapPercent, nfft)
    % Inputs:
    %   signal        - Input signal to be analyzed
    %   fs            - Sampling frequency (Hz)
    %   windowLength  - Length of each segment (samples)
    %   overlapPercent- Percentage of overlap between segments (0 to 100)
    %   nfft          - Number of FFT points, frequency resolution 

    % Outputs:
    %   fX            - Frequency axis values (Hz)
    %   tX            - Time axis values (s)
    %   specMatrix    - Spectrogram matrix with complex values

    % Prepare the signal
    y = single(signal(:));  % Convert to column vector and single-precision
    ylen = length(y);       % Get the signal length

    % Define the window function
    window = hann(windowLength);

    % Convert overlap percentage to number of overlapping samples
    overlap = floor(windowLength * overlapPercent / 100);

    % Calculate the number of segments
    numSegments = floor((ylen - overlap) / (windowLength - overlap));
    
    % Initialize the spectrogram matrix
    specMatrix = zeros(nfft, numSegments); % Full FFT range

    % Loop over each segment
    for i = 1:numSegments
        startIdx = (i-1) * (windowLength - overlap) + 1;    % Calculate start index for the segment
        endIdx = startIdx + windowLength - 1;               % Calculate end index for the segment
        segment = y(startIdx:endIdx) - mean(y(startIdx:endIdx)); % Remove DC component, Extract the segment
        windowedSegment = segment .* window;                % Apply windowing function
        spectrum = fft(windowedSegment, nfft);              % Compute FFT on the segment
        specMatrix(:, i) = spectrum;                        % Append FFT of the segment to the final result
    end

    % Define time and frequency vectors
    tX = (0:numSegments-1) * (windowLength - overlap) / fs;
    fX = (-nfft/2:nfft/2-1) * (fs / nfft);

    % Shift the zero-frequency component to the center of the spectrum
    specMatrix = fftshift(specMatrix, 1);
end
