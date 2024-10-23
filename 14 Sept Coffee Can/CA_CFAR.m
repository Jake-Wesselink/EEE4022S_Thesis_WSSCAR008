function [Detection] = CA_CFAR(spec, PFA, RefWindow, GaurdCells)

[K, L] = size(spec);  % K corresponds to speed bins (rows), L to time bins (columns)
abs_spec = abs(spec).^2;

% CA-CFAR implementation
N = RefWindow;  % Number of reference cells

Threshold = zeros(K, L);
Detection = zeros(K, L);
detection_count = 0;

%define cuts
StartCUTCell = N+GaurdCells+1;
StopCUTCell = K-N-GaurdCells;

% Iterate over time bins
for l = 1 : L  % Iterate over time instances (columns)
   
    for k = StartCUTCell : StopCUTCell  % Iterate over speed bins (rows)
        
        StartCellLaggingWindow = k-N-GaurdCells;
        StopCellLaggingWindow = k-GaurdCells-1;
    
        StartCellLeadingWindow = k+GaurdCells+1; 
        StopCellLeadingWindow = k+N+GaurdCells;
        
        RefCells = [abs_spec(StartCellLaggingWindow:StopCellLaggingWindow, l); ...
                    abs_spec(StartCellLeadingWindow:StopCellLeadingWindow, l)];

        Z = sum(RefCells) / (2 * N);  % Average of the reference cells
        alpha_CA = (2 * N) * (PFA^(-1/(2*N)) - 1);  % Scaling factor alpha
        Threshold(k, l) = alpha_CA * Z;  % Threshold value
        
        % Check if the current cell exceeds the threshold
        if abs_spec(k, l) > Threshold(k, l)
            Detection(k, l) = 1;  % Detect target
            detection_count = detection_count + 1;
        end
        
    end
end


% Calculate and display the simulated PFA
PFA_sim = detection_count / (K * L);

%disp(['Number of Detections: ', num2str(detection_count)]);
%disp(['PFA: ', num2str(PFA)]);
%disp(['PFA_simulation: ', num2str(PFA_sim)]);
% PFA_error = abs((PFA - PFA_sim) / PFA * 100);
% disp(['PFA_error: ', num2str(PFA_error), '%']); this calculation no
% longer makes sense as the target is being detected 

end
