% This function draws the precision curves from the mat files
figure(1)
plot(0:50,[0 mean(pr_deep_hsi2(:,1:50),1)],'g','LineWidth',2)
hold on
plot(0:50,[0 mean(pr_deep_rgb2(:,1:50),1)],'r','LineWidth',2)
plot(0:50,[0 mean(pr_hog(:,1:50),1)],'b','LineWidth',2)
plot(0:50,[0 mean(pr_hsi(:,1:50),1)],'k','LineWidth',2)
plot(0:50,[0 mean(pr_hsi_hog(:,1:50),1)],'c','LineWidth',2)
set(gcf,'Units','inches','Position',[0 0 2.5 2.5]);
axis([0 50 0 1]);
axis([0 50 0 0.85]);
axis([0 50 0 0.90]);
xlabel('Location Error Threshold')
ylabel('Precision');
print('precision_features','-depsc');
print('precision_features','-depsc');
set(gca,'Xtick',[0:10:50]);
set(gca,'Ytick',[0:0.1:0.90]);
print('precision_features','-depsc');