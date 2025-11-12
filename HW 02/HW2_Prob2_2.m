X = load("pclX.txt");
Y= load("pclY.txt");

transformed=[]

% R Matrix from prior script
R = [0.951266006558687	-0.150430576214007	-0.269190687999811;
    0.223236284835894	0.938163601035720	0.264602756645424;
    0.21274056006920	-0.311800736740008	0.926024705215704]

%T Matrix from prior script
t = [0.496614869133391
-0.293929711917010
0.296450043082626]

scatter3(X(:,1),X(:,2),X(:,3),5,'filled','red')
hold on
scatter3(Y(:,1),Y(:,2),Y(:,3),5,'filled','blue')

length = size(X,1)

for i=1:1:length
    xi = X(i,:)';
    trans = R*xi + t;
    transformed(i,:) = trans';
end 

scatter3(transformed(:,1),transformed(:,2),transformed(:,3),5,'filled','green')
hold on
scatter3(Y(:,1),Y(:,2),Y(:,3),5,'filled','blue')
legend ('Original','Baseline','Transformed')