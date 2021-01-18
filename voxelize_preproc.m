clear;
direction = "Y";
% direction = "Z";

%% Voxelize the main body
% Define bounding box
xmin1 = 0; xmax1 = 488;
ymin1 = 0; ymax1 = 488;
zmin1 = 0; zmax1 = 1000;
% Expand bounding box by [pad] mm
pad = 12;
xmax = xmax1 + pad; xmin = xmin1 - pad;
ymax = ymax1 + pad; ymin = ymin1 - pad;
zmax = zmax1 + pad; zmin = zmin1 - pad;
% Determine resolution
nelx = 128; nely = 128; nelz = 256;
dx = (xmax-xmin)/nelx;
dy = (ymax-ymin)/nely;
dz = (zmax-zmin)/nelz;
%% Voxelize the voids
nonvoids = zeros(nelx,nely,nelz);
nonvoids(1:nelx,1:nely,nelz-floor(pad/dz)+1:nelz) = 1; % This is the building plate
nonvoids(floor(pad/dx)+1:nelx-floor(pad/dx),floor(pad/dy)+1:nely-floor(pad/dy),floor(pad/dz)+1:nelz-floor(pad/dz)) = 1; % This is the design domain
nonvoids = nonvoids(:);
voids = 1. - nonvoids;
voids = voids(:);
PetscBinaryWrite('void.bin',voids);
%% Create the slices
switch direction 
    case "Y" % Create the slices - in Y direction
        totalsize = nely-2*floor(pad/dy);
        slices = [0 floor(pad/dy)+floor(totalsize/3) floor(pad/dy)+2*floor(totalsize/3) 2*floor(pad/dy)+totalsize];
        for slc = 2:size(slices,2)
            inslice = zeros(nelx,nely,nelz);
            inslice(:,slices(slc-1)+1:slices(slc),:) = 1;
            inslice = inslice(:).*nonvoids;
            PetscBinaryWrite(['SLC_' int2str(slc-1) '.bin'],inslice);
        end
    case "Z" % Create the slices - in Z direction
        totalsize = nelz-floor(pad/dz);
        for slc = 2:size(slices,2)
            inslice = zeros(nelx,nely,nelz);
            inslice(:,:,slices(slc-1)+1:slices(slc)) = 1;
            inslice = inslice(:).*nonvoids;
            PetscBinaryWrite(['SLC_' int2str(slc-1) '.bin'],inslice);
        end
end
%% Voxelize the solids
solids = zeros(nelx,nely,nelz);
% Load at Y=0, Z=0
solids(floor(pad/dx)+1:nelx-floor(pad/dx),floor(pad/dy)+1:2*floor(pad/dy),floor(pad/dz)+1:2*floor(pad/dz)) = 1; % This is the load at y=0
solids = solids(:);
PetscBinaryWrite('solid.bin',solids);
%% Boundary conditions
% The wall for regular loadcases
supportnodes = ones(nelx+1,nely+1,nelz+1);
% Wall at Z=Zmax
supportnodes(floor(pad/dx)+1:nelx+1-floor(pad/dx),floor(pad/dy)+1:nely+1-floor(pad/dy),nelz+1-floor(pad/dz):nelz+1-floor(pad/dz)) = 0;
supportnodes = supportnodes(:);
supportdofs = repmat(supportnodes',3,1);
supportdofs = supportdofs(:);
PetscBinaryWrite('BC_1.bin',supportdofs);  
% The building plate
switch direction 
    case "Y"
        supportnodes = ones(nelx+1,nely+1,nelz+1);
        supportnodes(1:nelx+1,floor(pad/dy)+1,1:nelz+1) = 0;
    case "Z"
        supportnodes = ones(nelx+1,nely+1,nelz+1);
        supportnodes(1:nelx+1,1:nely+1,1:floor(pad/dz)+1) = 0;
end
supportnodes = supportnodes(:);
supportdofs = repmat(supportnodes',3,1);
supportdofs = supportdofs(:);
PetscBinaryWrite('BCstg.bin',supportdofs);  
%% Loadcases
loadnodes = zeros(nelx+1,nely+1,nelz+1);
% % Load at Y=0, Z=0
loadnodes(floor(pad/dx)+1:nelx+1-floor(pad/dx),floor(pad/dy)+1:2*floor(pad/dy)+1,floor(pad/dz)+1:2*floor(pad/dz)+1) = 1; % at y=0
loadnodes = loadnodes(:);
loaddofs = repmat(loadnodes',3,1);
loaddofs(1,:) = 0*loaddofs(1,:);
loaddofs(2,:) = -1000*loaddofs(2,:);
loaddofs(3,:) = 0*loaddofs(3,:);
loaddofs = loaddofs(:);
RHS1 = loaddofs;
PetscBinaryWrite('RHS_1_1_1.bin',RHS1);

