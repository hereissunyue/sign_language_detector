%% Preface
% Name: Mahdieh Nejati Javaremi     ID:              Email:m.nejati@u.northwestern.edu
% Name: Yue Sun                     ID: 2880445      Email:sunyue@u.northwestern.edu
% Final Project: Sign Language Recognition with Machine Learning Algorithm
% Description: Thie .m file is used for image processing test demonstrating processing details
close all


%% Setting the path and Reading the Images
% Dear Professor and TA, please change this path corresponding to your computer! Thank you!
addpath('C:\Users\Mahdieh\OneDrive\Documents\MSR Spring Quarter 2015\machine learning\Project\sign_language_detector\library')  % path of library
addpath('C:\Users\Mahdieh\OneDrive\Documents\MSR Spring Quarter 2015\machine learning\Project\sign_language_detector\Data') % path of image

% Reading the test images
A1img = imread('A1.jpg'); A1img = imresize(A1img, 0.25);
A2img = imread('A2.jpg'); A2img = imresize(A2img, 0.25);
A3img = imread('A3.jpg'); A3img = imresize(A3img, 0.25);
A4img = imread('A4.jpg'); A4img = imresize(A4img, 0.25);
A5img = imread('A5.jpg'); A5img = imresize(A5img, 0.25);
A6img = imread('A6.jpg'); A6img = imresize(A6img, 0.25);
A7img = imread('A7.jpg'); A7img = imresize(A7img, 0.25);
A8img = imread('A8.jpg'); A8img = imresize(A8img, 0.25);
A9img = imread('A9.jpg'); A9img = imresize(A9img, 0.25);
A10img = imread('A10.jpg'); A10img = imresize(A10img, 0.25);

B1img = imread('B1.jpg'); B1img = imresize(B1img, 0.25);
B2img = imread('B2.jpg'); B2img = imresize(B2img, 0.25);
B3img = imread('B3.jpg'); B3img = imresize(B3img, 0.25);
B4img = imread('B4.jpg'); B4img = imresize(B4img, 0.25);
B5img = imread('B5.jpg'); B5img = imresize(B5img, 0.25);
B6img = imread('B6.jpg'); B6img = imresize(B6img, 0.25);
B7img = imread('B7.jpg'); B7img = imresize(B7img, 0.25);
B8img = imread('B8.jpg'); B8img = imresize(B8img, 0.25);
B9img = imread('B9.jpg'); B9img = imresize(B9img, 0.25);
B10img = imread('B10.jpg'); B10img = imresize(B10img, 0.25);

C1img = imread('C1.jpg'); C1img = imresize(C1img, 0.25);
C2img = imread('C2.jpg'); C2img = imresize(C2img, 0.25);
C3img = imread('C3.jpg'); C3img = imresize(C3img, 0.25);
C4img = imread('C4.jpg'); C4img = imresize(C4img, 0.25);
C5img = imread('C5.jpg'); C5img = imresize(C5img, 0.25);
C6img = imread('C6.jpg'); C6img = imresize(C6img, 0.25);
C7img = imread('C7.jpg'); C7img = imresize(C7img, 0.25);
C8img = imread('C8.jpg'); C8img = imresize(C8img, 0.25);
C9img = imread('C9.jpg'); C9img = imresize(C9img, 0.25);
C10img = imread('C10.jpg'); C10img = imresize(C10img, 0.25);

D1img = imread('D1.jpg'); D1img = imresize(D1img, 0.25);
D2img = imread('D2.jpg'); D2img = imresize(D2img, 0.25);
D3img = imread('D3.jpg'); D3img = imresize(D3img, 0.25);
D4img = imread('D4.jpg'); D4img = imresize(D4img, 0.25);
D5img = imread('D5.jpg'); D5img = imresize(D5img, 0.25);
D6img = imread('D6.jpg'); D6img = imresize(D6img, 0.25);
D7img = imread('D7.jpg'); D7img = imresize(D7img, 0.25);
D8img = imread('D8.jpg'); D8img = imresize(D8img, 0.25);
D9img = imread('D9.jpg'); D9img = imresize(D9img, 0.25);
D10img = imread('D10.jpg'); D10img = imresize(D10img, 0.25);

E1img = imread('E1.jpg'); E1img = imresize(E1img, 0.25);
E2img = imread('E2.jpg'); E2img = imresize(E2img, 0.25);
E3img = imread('E3.jpg'); E3img = imresize(E3img, 0.25);
E4img = imread('E4.jpg'); E4img = imresize(E4img, 0.25);
E5img = imread('E5.jpg'); E5img = imresize(E5img, 0.25);
E6img = imread('E6.jpg'); E6img = imresize(E6img, 0.25);
E7img = imread('E7.jpg'); E7img = imresize(E7img, 0.25);
E8img = imread('E8.jpg'); E8img = imresize(E8img, 0.25);
E9img = imread('E9.jpg'); E9img = imresize(E9img, 0.25);
E10img = imread('E10.jpg'); E10img = imresize(E10img, 0.25);

F1img = imread('F1.jpg'); F1img = imresize(F1img, 0.25);
F2img = imread('F2.jpg'); F2img = imresize(F2img, 0.25);
F3img = imread('F3.jpg'); F3img = imresize(F3img, 0.25);
F4img = imread('F4.jpg'); F4img = imresize(F4img, 0.25);
F5img = imread('F5.jpg'); F5img = imresize(F5img, 0.25);
F6img = imread('F6.jpg'); F6img = imresize(F6img, 0.25);
F7img = imread('F7.jpg'); F7img = imresize(F7img, 0.25);
F8img = imread('F8.jpg'); F8img = imresize(F8img, 0.25);
F9img = imread('F9.jpg'); F9img = imresize(F9img, 0.25);
F10img = imread('F10.jpg'); F10img = imresize(F10img, 0.25);

G1img = imread('G1.jpg'); G1img = imresize(G1img, 0.25);
G2img = imread('G2.jpg'); G2img = imresize(G2img, 0.25);
G3img = imread('G3.jpg'); G3img = imresize(G3img, 0.25);
G4img = imread('G4.jpg'); G4img = imresize(G4img, 0.25);
G5img = imread('G5.jpg'); G5img = imresize(G5img, 0.25);
G6img = imread('G6.jpg'); G6img = imresize(G6img, 0.25);
G7img = imread('G7.jpg'); G7img = imresize(G7img, 0.25);
G8img = imread('G8.jpg'); G8img = imresize(G8img, 0.25);
G9img = imread('G9.jpg'); G9img = imresize(G9img, 0.25);
G10img = imread('G10.jpg'); G10img = imresize(G10img, 0.25);

H1img = imread('H1.jpg'); H1img = imresize(H1img, 0.25);
H2img = imread('H2.jpg'); H2img = imresize(H2img, 0.25);
H3img = imread('H3.jpg'); H3img = imresize(H3img, 0.25);
H4img = imread('H4.jpg'); H4img = imresize(H4img, 0.25);
H5img = imread('H5.jpg'); H5img = imresize(H5img, 0.25);
H6img = imread('H6.jpg'); H6img = imresize(H6img, 0.25);
H7img = imread('H7.jpg'); H7img = imresize(H7img, 0.25);
H8img = imread('H8.jpg'); H8img = imresize(H8img, 0.25);
H9img = imread('H9.jpg'); H9img = imresize(H9img, 0.25);
H10img = imread('H10.jpg'); H10img = imresize(H10img, 0.25);

I1img = imread('I1.jpg'); I1img = imresize(I1img, 0.25);
I2img = imread('I2.jpg'); I2img = imresize(I2img, 0.25);
I3img = imread('I3.jpg'); I3img = imresize(I3img, 0.25);
I4img = imread('I4.jpg'); I4img = imresize(I4img, 0.25);
I5img = imread('I5.jpg'); I5img = imresize(I5img, 0.25);
I6img = imread('I6.jpg'); I6img = imresize(I6img, 0.25);
I7img = imread('I7.jpg'); I7img = imresize(I7img, 0.25);
I8img = imread('I8.jpg'); I8img = imresize(I8img, 0.25);
I9img = imread('I9.jpg'); I9img = imresize(I9img, 0.25);
I10img = imread('I10.jpg'); I10img = imresize(I10img, 0.25);

J1img = imread('J1.jpg'); J1img = imresize(J1img, 0.25);
J2img = imread('J2.jpg'); J2img = imresize(J2img, 0.25);
J3img = imread('J3.jpg'); J3img = imresize(J3img, 0.25);
J4img = imread('J4.jpg'); J4img = imresize(J4img, 0.25);
J5img = imread('J5.jpg'); J5img = imresize(J5img, 0.25);
J6img = imread('J6.jpg'); J6img = imresize(J6img, 0.25);
J7img = imread('J7.jpg'); J7img = imresize(J7img, 0.25);
J8img = imread('J8.jpg'); J8img = imresize(J8img, 0.25);
J9img = imread('J9.jpg'); J9img = imresize(J9img, 0.25);
J10img = imread('J10.jpg'); J10img = imresize(J10img, 0.25);

K1img = imread('K1.jpg'); K1img = imresize(K1img, 0.25);
K2img = imread('K2.jpg'); K2img = imresize(K2img, 0.25);
K3img = imread('K3.jpg'); K3img = imresize(K3img, 0.25);
K4img = imread('K4.jpg'); K4img = imresize(K4img, 0.25);
K5img = imread('K5.jpg'); K5img = imresize(K5img, 0.25);
K6img = imread('K6.jpg'); K6img = imresize(K6img, 0.25);
K7img = imread('K7.jpg'); K7img = imresize(K7img, 0.25);
K8img = imread('K8.jpg'); K8img = imresize(K8img, 0.25);
K9img = imread('K9.jpg'); K9img = imresize(K9img, 0.25);
K10img = imread('K10.jpg'); K10img = imresize(K10img, 0.25);

L1img = imread('L1.jpg'); L1img = imresize(L1img, 0.25);
L2img = imread('L2.jpg'); L2img = imresize(L2img, 0.25);
L3img = imread('L3.jpg'); L3img = imresize(L3img, 0.25);
L4img = imread('L4.jpg'); L4img = imresize(L4img, 0.25);
L5img = imread('L5.jpg'); L5img = imresize(L5img, 0.25);
L6img = imread('L6.jpg'); L6img = imresize(L6img, 0.25);
L7img = imread('L7.jpg'); L7img = imresize(L7img, 0.25);
L8img = imread('L8.jpg'); L8img = imresize(L8img, 0.25);
L9img = imread('L9.jpg'); L9img = imresize(L9img, 0.25);
L10img = imread('L10.jpg'); L10img = imresize(L10img, 0.25);

M1img = imread('M1.jpg'); M1img = imresize(M1img, 0.25);
M2img = imread('M2.jpg'); M2img = imresize(M2img, 0.25);
M3img = imread('M3.jpg'); M3img = imresize(M3img, 0.25);
M4img = imread('M4.jpg'); M4img = imresize(M4img, 0.25);
M5img = imread('M5.jpg'); M5img = imresize(M5img, 0.25);
M6img = imread('M6.jpg'); M6img = imresize(M6img, 0.25);
M7img = imread('M7.jpg'); M7img = imresize(M7img, 0.25);
M8img = imread('M8.jpg'); M8img = imresize(M8img, 0.25);
M9img = imread('M9.jpg'); M9img = imresize(M9img, 0.25);
M10img = imread('M10.jpg'); M10img = imresize(M10img, 0.25);

N1img = imread('N1.jpg'); N1img = imresize(N1img, 0.25);
N2img = imread('N2.jpg'); N2img = imresize(N2img, 0.25);
N3img = imread('N3.jpg'); N3img = imresize(N3img, 0.25);
N4img = imread('N4.jpg'); N4img = imresize(N4img, 0.25);
N5img = imread('N5.jpg'); N5img = imresize(N5img, 0.25);
N6img = imread('N6.jpg'); N6img = imresize(N6img, 0.25);
N7img = imread('N7.jpg'); N7img = imresize(N7img, 0.25);
N8img = imread('N8.jpg'); N8img = imresize(N8img, 0.25);
N9img = imread('N9.jpg'); N9img = imresize(N9img, 0.25);
N10img = imread('N10.jpg'); N10img = imresize(N10img, 0.25);

O1img = imread('O1.jpg'); O1img = imresize(O1img, 0.25);
O2img = imread('O2.jpg'); O2img = imresize(O2img, 0.25);
O3img = imread('O3.jpg'); O3img = imresize(O3img, 0.25);
O4img = imread('O4.jpg'); O4img = imresize(O4img, 0.25);
O5img = imread('O5.jpg'); O5img = imresize(O5img, 0.25);
O6img = imread('O6.jpg'); O6img = imresize(O6img, 0.25);
O7img = imread('O7.jpg'); O7img = imresize(O7img, 0.25);
O8img = imread('O8.jpg'); O8img = imresize(O8img, 0.25);
O9img = imread('O9.jpg'); O9img = imresize(O9img, 0.25);
O10img = imread('O10.jpg'); O10img = imresize(O10img, 0.25);

P1img = imread('P1.jpg'); P1img = imresize(P1img, 0.25);
P2img = imread('P2.jpg'); P2img = imresize(P2img, 0.25);
P3img = imread('P3.jpg'); P3img = imresize(P3img, 0.25);
P4img = imread('P4.jpg'); P4img = imresize(P4img, 0.25);
P5img = imread('P5.jpg'); P5img = imresize(P5img, 0.25);
P6img = imread('P6.jpg'); P6img = imresize(P6img, 0.25);
P7img = imread('P7.jpg'); P7img = imresize(P7img, 0.25);
P8img = imread('P8.jpg'); P8img = imresize(P8img, 0.25);
P9img = imread('P9.jpg'); P9img = imresize(P9img, 0.25);
P10img = imread('P10.jpg'); P10img = imresize(P10img, 0.25);

Q1img = imread('Q1.jpg'); Q1img = imresize(Q1img, 0.25);
Q2img = imread('Q2.jpg'); Q2img = imresize(Q2img, 0.25);
Q3img = imread('Q3.jpg'); Q3img = imresize(Q3img, 0.25);
Q4img = imread('Q4.jpg'); Q4img = imresize(Q4img, 0.25);
Q5img = imread('Q5.jpg'); Q5img = imresize(Q5img, 0.25);
Q6img = imread('Q6.jpg'); Q6img = imresize(Q6img, 0.25);
Q7img = imread('Q7.jpg'); Q7img = imresize(Q7img, 0.25);
Q8img = imread('Q8.jpg'); Q8img = imresize(Q8img, 0.25);
Q9img = imread('Q9.jpg'); Q9img = imresize(Q9img, 0.25);
Q10img = imread('Q10.jpg'); Q10img = imresize(Q10img, 0.25);

R1img = imread('R1.jpg'); R1img = imresize(R1img, 0.25);
R2img = imread('R2.jpg'); R2img = imresize(R2img, 0.25);
R3img = imread('R3.jpg'); R3img = imresize(R3img, 0.25);
R4img = imread('R4.jpg'); R4img = imresize(R4img, 0.25);
R5img = imread('R5.jpg'); R5img = imresize(R5img, 0.25);
R6img = imread('R6.jpg'); R6img = imresize(R6img, 0.25);
R7img = imread('R7.jpg'); R7img = imresize(R7img, 0.25);
R8img = imread('R8.jpg'); R8img = imresize(R8img, 0.25);
R9img = imread('R9.jpg'); R9img = imresize(R9img, 0.25);
R10img = imread('R10.jpg'); R10img = imresize(R10img, 0.25);

S1img = imread('S1.jpg'); S1img = imresize(S1img, 0.25);
S2img = imread('S2.jpg'); S2img = imresize(S2img, 0.25);
S3img = imread('S3.jpg'); S3img = imresize(S3img, 0.25);
S4img = imread('S4.jpg'); S4img = imresize(S4img, 0.25);
S5img = imread('S5.jpg'); S5img = imresize(S5img, 0.25);
S6img = imread('S6.jpg'); S6img = imresize(S6img, 0.25);
S7img = imread('S7.jpg'); S7img = imresize(S7img, 0.25);
S8img = imread('S8.jpg'); S8img = imresize(S8img, 0.25);
S9img = imread('S9.jpg'); S9img = imresize(S9img, 0.25);
S10img = imread('S10.jpg'); S10img = imresize(S10img, 0.25);

T1img = imread('T1.jpg'); T1img = imresize(T1img, 0.25);
T2img = imread('T2.jpg'); T2img = imresize(T2img, 0.25);
T3img = imread('T3.jpg'); T3img = imresize(T3img, 0.25);
T4img = imread('T4.jpg'); T4img = imresize(T4img, 0.25);
T5img = imread('T5.jpg'); T5img = imresize(T5img, 0.25);
T6img = imread('T6.jpg'); T6img = imresize(T6img, 0.25);
T7img = imread('T7.jpg'); T7img = imresize(T7img, 0.25);
T8img = imread('T8.jpg'); T8img = imresize(T8img, 0.25);
T9img = imread('T9.jpg'); T9img = imresize(T9img, 0.25);
T10img = imread('T10.jpg'); T10img = imresize(T10img, 0.25);

U1img = imread('U1.jpg'); U1img = imresize(U1img, 0.25);
U2img = imread('U2.jpg'); U2img = imresize(U2img, 0.25);
U3img = imread('U3.jpg'); U3img = imresize(U3img, 0.25);
U4img = imread('U4.jpg'); U4img = imresize(U4img, 0.25);
U5img = imread('U5.jpg'); U5img = imresize(U5img, 0.25);
U6img = imread('U6.jpg'); U6img = imresize(U6img, 0.25);
U7img = imread('U7.jpg'); U7img = imresize(U7img, 0.25);
U8img = imread('U8.jpg'); U8img = imresize(U8img, 0.25);
U9img = imread('U9.jpg'); U9img = imresize(U9img, 0.25);
U10img = imread('U10.jpg'); U10img = imresize(U10img, 0.25);

V1img = imread('V1.jpg'); V1img = imresize(V1img, 0.25);
V2img = imread('V2.jpg'); V2img = imresize(V2img, 0.25);
V3img = imread('V3.jpg'); V3img = imresize(V3img, 0.25);
V4img = imread('V4.jpg'); V4img = imresize(V4img, 0.25);
V5img = imread('V5.jpg'); V5img = imresize(V5img, 0.25);
V6img = imread('V6.jpg'); V6img = imresize(V6img, 0.25);
V7img = imread('V7.jpg'); V7img = imresize(V7img, 0.25);
V8img = imread('V8.jpg'); V8img = imresize(V8img, 0.25);
V9img = imread('V9.jpg'); V9img = imresize(V9img, 0.25);
V10img = imread('V10.jpg'); V10img = imresize(V10img, 0.25);

W1img = imread('W1.jpg'); W1img = imresize(W1img, 0.25);
W2img = imread('W2.jpg'); W2img = imresize(W2img, 0.25);
W3img = imread('W3.jpg'); W3img = imresize(W3img, 0.25);
W4img = imread('W4.jpg'); W4img = imresize(W4img, 0.25);
W5img = imread('W5.jpg'); W5img = imresize(W5img, 0.25);
W6img = imread('W6.jpg'); W6img = imresize(W6img, 0.25);
W7img = imread('W7.jpg'); W7img = imresize(W7img, 0.25);
W8img = imread('W8.jpg'); W8img = imresize(W8img, 0.25);
W9img = imread('W9.jpg'); W9img = imresize(W9img, 0.25);
W10img = imread('W10.jpg'); W10img = imresize(W10img, 0.25);

X1img = imread('X1.jpg'); X1img = imresize(X1img, 0.25);
X2img = imread('X2.jpg'); X2img = imresize(X2img, 0.25);
X3img = imread('X3.jpg'); X3img = imresize(X3img, 0.25);
X4img = imread('X4.jpg'); X4img = imresize(X4img, 0.25);
X5img = imread('X5.jpg'); X5img = imresize(X5img, 0.25);
X6img = imread('X6.jpg'); X6img = imresize(X6img, 0.25);
X7img = imread('X7.jpg'); X7img = imresize(X7img, 0.25);
X8img = imread('X8.jpg'); X8img = imresize(X8img, 0.25);
X9img = imread('X9.jpg'); X9img = imresize(X9img, 0.25);
X10img = imread('X10.jpg'); X10img = imresize(X10img, 0.25);

Y1img = imread('Y1.jpg'); Y1img = imresize(Y1img, 0.25);
Y2img = imread('Y2.jpg'); Y2img = imresize(Y2img, 0.25);
Y3img = imread('Y3.jpg'); Y3img = imresize(Y3img, 0.25);
Y4img = imread('Y4.jpg'); Y4img = imresize(Y4img, 0.25);
Y5img = imread('Y5.jpg'); Y5img = imresize(Y5img, 0.25);
Y6img = imread('Y6.jpg'); Y6img = imresize(Y6img, 0.25);
Y7img = imread('Y7.jpg'); Y7img = imresize(Y7img, 0.25);
Y8img = imread('Y8.jpg'); Y8img = imresize(Y8img, 0.25);
Y9img = imread('Y9.jpg'); Y9img = imresize(Y9img, 0.25);
Y10img = imread('Y10.jpg'); Y10img = imresize(Y10img, 0.25);

Z1img = imread('Z1.jpg'); Z1img = imresize(Z1img, 0.25);
Z2img = imread('Z2.jpg'); Z2img = imresize(Z2img, 0.25);
Z3img = imread('Z3.jpg'); Z3img = imresize(Z3img, 0.25);
Z4img = imread('Z4.jpg'); Z4img = imresize(Z4img, 0.25);
Z5img = imread('Z5.jpg'); Z5img = imresize(Z5img, 0.25);
Z6img = imread('Z6.jpg'); Z6img = imresize(Z6img, 0.25);
Z7img = imread('Z7.jpg'); Z7img = imresize(Z7img, 0.25);
Z8img = imread('Z8.jpg'); Z8img = imresize(Z8img, 0.25);
Z9img = imread('Z9.jpg'); Z9img = imresize(Z9img, 0.25);
Z10img = imread('Z10.jpg'); Z10img = imresize(Z10img, 0.25);

% save the data in a cell for convinence
sampleimg = {A1img,A2img,A31img,A41img,A5img,A6img,A7img,A8img,A9img,A10img,
    B1img,B2img,B31img,B41img,B5img,B6img,B7img,B8img,B9img,B10img,
    C1img,C2img,C31img,C41img,C5img,C6img,C7img,C8img,C9img,C10img,
    D1img,D2img,D31img,D41img,D5img,D6img,D7img,D8img,D9img,D10img,
    E1img,E2img,E31img,E41img,E5img,E6img,E7img,E8img,E9img,E10img,
    F1img,F2img,F31img,F41img,F5img,F6img,F7img,F8img,F9img,F10img,
    G1img,G2img,G31img,G41img,G5img,G6img,G7img,G8img,G9img,G10img,
    H1img,H2img,H31img,H41img,H5img,H6img,H7img,H8img,H9img,H10img,
    I1img,I2img,I31img,I41img,I5img,I6img,I7img,I8img,I9img,I10img,
    J1img,J2img,J31img,J41img,J5img,J6img,J7img,J8img,J9img,J10img,
    K1img,K2img,K31img,K41img,K5img,K6img,K7img,K8img,K9img,K10img,
    L1img,L2img,L31img,L41img,L5img,L6img,L7img,L8img,L9img,L10img,
    M1img,M2img,M31img,M41img,M5img,M6img,M7img,M8img,M9img,M10img,
    N1img,N2img,N31img,N41img,N5img,N6img,N7img,N8img,N9img,N10img,
    O1img,O2img,O31img,O41img,O5img,O6img,O7img,O8img,O9img,O10img,
    P1img,P2img,P31img,P41img,P5img,P6img,P7img,P8img,P9img,P10img,
    Q1img,Q2img,Q31img,Q41img,Q5img,Q6img,Q7img,Q8img,Q9img,Q10img,
    R1img,R2img,R31img,R41img,R5img,R6img,R7img,R8img,R9img,R10img,
    S1img,S2img,S31img,S41img,S5img,S6img,S7img,S8img,S9img,S10img,
    T1img,T2img,T31img,T41img,T5img,T6img,T7img,T8img,T9img,T10img,
    U1img,U2img,U31img,U41img,U5img,U6img,U7img,U8img,U9img,U10img,
    V1img,V2img,V31img,V41img,V5img,V6img,V7img,V8img,V9img,V10img,
    W1img,W2img,W31img,W41img,W5img,W6img,W7img,W8img,W9img,W10img,
    X1img,X2img,X31img,X41img,X5img,X6img,X7img,X8img,X9img,X10img,
    Y1img,Y2img,Y31img,Y41img,Y5img,Y6img,Y7img,Y8img,Y9img,Y10img,
    Z1img,Z2img,Z31img,Z41img,Z5img,Z6img,Z7img,Z8img,Z9img,Z10img};

SignClass = {'A'; 'A'; 'A'; 'A'; 'A'; 'A'; 'A'; 'A'; 'A'; 'A';
    'B'; 'B'; 'B'; 'B'; 'B'; 'B'; 'B'; 'B'; 'B'; 'B';
    'C'; 'C'; 'C'; 'C'; 'C'; 'C'; 'C'; 'C'; 'C'; 'C';
    'D'; 'D'; 'D'; 'D'; 'D'; 'D'; 'D'; 'D'; 'D'; 'D';
    'E'; 'E'; 'E'; 'E'; 'E'; 'E'; 'E'; 'E'; 'E'; 'E';
    'F'; 'F'; 'F'; 'F'; 'F'; 'F'; 'F'; 'F'; 'F'; 'F';
    'G'; 'G'; 'G'; 'G'; 'G'; 'G'; 'G'; 'G'; 'G'; 'G';
    'H'; 'H'; 'H'; 'H'; 'H'; 'H'; 'H'; 'H'; 'H'; 'H';
    'I'; 'I'; 'I'; 'I'; 'I'; 'I'; 'I'; 'I'; 'I'; 'I';
    'J'; 'J'; 'J'; 'J'; 'J'; 'J'; 'J'; 'J'; 'J'; 'J';
    'K'; 'K'; 'K'; 'K'; 'K'; 'K'; 'K'; 'K'; 'K'; 'K';
    'L'; 'L'; 'L'; 'L'; 'L'; 'L'; 'L'; 'L'; 'L'; 'L';
    'M'; 'M'; 'M'; 'M'; 'M'; 'M'; 'M'; 'M'; 'M'; 'M';
    'N'; 'N'; 'N'; 'N'; 'N'; 'N'; 'N'; 'N'; 'N'; 'N';
    'O'; 'O'; 'O'; 'O'; 'O'; 'O'; 'O'; 'O'; 'O'; 'O';
    'P'; 'P'; 'P'; 'P'; 'P'; 'P'; 'P'; 'P'; 'P'; 'P';
    'Q'; 'Q'; 'Q'; 'Q'; 'Q'; 'Q'; 'Q'; 'Q'; 'Q'; 'Q';
    'R'; 'R'; 'R'; 'R'; 'R'; 'R'; 'R'; 'R'; 'R'; 'R';
    'S'; 'S'; 'S'; 'S'; 'S'; 'S'; 'S'; 'S'; 'S'; 'S';
    'T'; 'T'; 'T'; 'T'; 'T'; 'T'; 'T'; 'T'; 'T'; 'T';
    'U'; 'U'; 'U'; 'U'; 'U'; 'U'; 'U'; 'U'; 'U'; 'U';
    'V'; 'V'; 'V'; 'V'; 'V'; 'V'; 'V'; 'V'; 'V'; 'V';
    'W'; 'W'; 'W'; 'W'; 'W'; 'W'; 'W'; 'W'; 'W'; 'W';
    'X'; 'X'; 'X'; 'X'; 'X'; 'X'; 'X'; 'X'; 'X'; 'X';
    'Y'; 'Y'; 'Y'; 'Y'; 'Y'; 'Y'; 'Y'; 'Y'; 'Y'; 'Y';
    'Z'; 'Z'; 'Z'; 'Z'; 'Z'; 'Z'; 'Z'; 'Z'; 'Z'; 'Z'};

% % image showing
% for i = 1 : size(sampleimg,2)
%     figure(i)
%     imshow(sampleimg{i})
% end


%% Preprocessing the image
% The goal of the preprocessing

% Processing a certain image
num = 2;
Img = sampleimg{num};
ImgGray= rgb2gray(Img); % Transfer the RGB to Gray
% HSV filter in order to eliminate the back ground
ImgHSV = HSVfilter(Img, [0.6186, 0.6657, 0.1865], [0.3 0.3 0.4]);
% Erosion and Dilation parameters
seD = strel('diamond',3);
se90 = strel('line', 4, 90);
se0 = strel('line', 4, 0);
ImgDi = imdilate(ImgHSV, [se90 se0]); % Dilation for eliminating noise
ImgEr = imerode(ImgDi,seD); % Erosion to get back the shape area
ImgBoundaryEdge = edge(ImgEr,'Prewitt'); % Edge of the boundary
[wristpoint, angle, Orientation] = ExtractHand(ImgBoundaryEdge); % Extract the hand information
[Connectlist] = ConnectPoints(wristpoint); % Connection list based on wrist point
ImgEr = imrotate(ImgEr,Orientation); % rotate the erosion image
[ImgErconnect] = Drawconnection(ImgEr, Connectlist); % Draw the connection on the Erosion image
Imginfo = regionprops(~ImgErconnect); % Extract the information
[area1,are2] = Imginfo.Area; % Access the area information
ImgHand = ~bwareaopen(~ImgErconnect, 2*min([area1,are2])); % small blob elimination

% Rotate the Target region
Irot = imrotate(ImgHand,angle);
Mrot = ~imrotate(true(size(ImgHand)),angle);
Irot(Mrot&~imclearborder(Mrot)) = 255;
ImgHand = Irot; % fill in white
[xmin,xmax,ymin,ymax] = Corpinfo(ImgHand); % the crop coordinate
ImgHand = ImgHand(xmin:xmax,ymin:ymax); % crop the ImgHand
ImgHand = imresize(ImgHand, [230 380]);
Img = imrotate(Img,angle+Orientation); % rotate the RGB image
Img = Img(xmin:xmax,ymin:ymax,1:3); % crop the Img
Img = imresize(Img, [230 380]);
ImgCorner = imrotate(ImgBoundaryEdge,angle+Orientation); % rotate the BoundaryEdge image
ImgCorner = ImgCorner(xmin:xmax,ymin:ymax); % crop the Img
ImgCorner = imresize(ImgCorner, [230 380]);
ImgHand = imdilate(ImgHand, [se90 se0]); % Dilation for eliminating noise
ImgHand = imerode(ImgHand,seD); % Erosion to get back the shape area
ImgBoundaryEdge = edge(ImgHand,'Prewitt'); % Edge of the boundary

% % Noisy Edge
ImgEdge = edge(rgb2gray(Img),'Prewitt',0.03); % Edge detail
ImgEdge = (~imerode(ImgHand,seD) & ImgEdge); % Background subtraction
se90 = strel('line', 2, 90);
se0 = strel('line', 2, 0);
seD = strel('diamond',1);
ImgEdge = imdilate(ImgEdge, [se90 se0]); % Dilation for eliminating noise
ImgEdge = imerode(ImgEdge,seD); % Erosion to get back the shape area
ImgEdge = bwareaopen(ImgEdge, 200); % small blob elimination

figure(1)
imshow(ImgHand)
figure(2)
imshow(Img)

figure(7)
imshow(ImgCorner)
figure(8)
imshow(ImgEdge)

figure(3)
C = corner(ImgCorner,500);
imshow(ImgCorner);
hold on
plot(C(:,1), C(:,2), 'r*');

[Gx,Gy] = imgradientxy(rgb2gray(Img));
figure(4)
imshow(Gx)
figure(5)
imshow(Gy)

figure(6)
imshow(rgb2gray(Img))
[featureVector, hogVisualization] = extractHOGFeatures((~imerode(ImgHand,seD))&rgb2gray(Img));
hold on
plot(hogVisualization);



figure(12)
ImgHand = imresize(ImgHand, [38 22]);
imshow(ImgHand)