function updatingBetaPostAndBernoulliSource(p1,p2)
addpath(genpath(pwd),'-begin') % top of search path in case of conflicting dependencies
% press any key to see next update
close
set(0,'defaultAxesFontSize',24)
figure; set(gcf,'units','normalized','OuterPosition',[0.1839 0 0.1814 0.8898])
ax(1) = subplot(2,1,1);
ax(2) = subplot(2,1,2);
D1 = binornd(1,p1,256,1); % 200 draws from a Bernoulli distribution
% with parameter p1
D2 = binornd(1,p2,256,1); % ditto but w parameter p2
theta0 = [.5 .5];

for d = [2 4 8 16 32 64 128 256]
    [Pest1,theta1,Pdist1,Pest2,theta2,Pdist2] = BayesEstBernP(D1(1:d),theta0,D2(1:d));

    plot(ax(1),Pdist1(:,1),Pdist1(:,2),'k-',Pdist2(:,1),Pdist2(:,2),'b-','LineWidth',2)
    % the two posteriors on same axes
    xlabel(ax(1),'\itp');ylabel(ax(1),'Likelihood')
    title(ax(1),'Beta Posteriors','FontWeight','normal')
    legend(ax(1),'p1','p2','Location','northwest')
    Ylm = ylim(ax(1));
    text(ax(1),.05,.5*Ylm(2),['\itn\rm = ' num2str(d)],'FontSize',24)
    text(ax(1),.05,.38*Ylm(2),['\theta_1=[' num2str(theta1(1)) ...
        ' ' num2str(theta1(2)) ']'],'FontSize',24)
    text(ax(1),.05,.26*Ylm(2),['\theta_2=[' num2str(theta2(1)) ' ' ...
        num2str(theta2(2)) ']'],'FontSize',24)
    %hold(ax(1),'off')
    
    h(1) = bar(ax(2),[1 2],[1-Pest1 Pest1],.8,'FaceColor',[0 0 0]);
    hold(ax(2),'on')
    h(2) = bar(ax(2),[4 5],[1-Pest2 Pest2],.8,'FaceColor','b');
    set(gca,'xtick',[1 2 4 5],'xticklabel',{'0' '1' '0' '1'})
    ylabel(ax(2),'Probability')
    xlabel(ax(2),'Bernoulli Support')
    title(ax(2),'Bernoulli Source Distributions','FontWeight','normal')
    hold(ax(2),'off')
    sprintf('\npress any key to see next update')
    pause
end

