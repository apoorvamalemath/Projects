#include<stdio.h>
#include<stdlib.h>

int main()
{
 int i,j,sum=0;
 float awt=0,atat=0;
 int n;
 printf("Enter the number of processes: \n");
 scanf("%d",&n);
 int wt[n],tat[n];
 int arrtime[n];
 int bursttime[n];
 int prio[n];
 printf("Enter the arrival time: \n");
 for(i=0;i<n;i++)
 scanf("%d",&arrtime[i]);
 printf("Enter the priority: \n");
 for(i=0;i<n;i++)
 scanf("%d",&prio[i]);

 printf("Enter the burst time: \n");
 for(i=0;i<n;i++)
 scanf("%d",&bursttime[i]);

 for(i=0;i<n-1;i++)
 {
   for(j=0;j<n-i-1;j++)
    if(prio[j]>prio[j+1])
     {
       int t,x;
       t=arrtime[j];
       arrtime[j]=arrtime[j+1];
       arrtime[j+1]=t;
       x=bursttime[j];
       bursttime[j]=bursttime[j+1];
       bursttime[j+1]=x;
     }

 }

 wt[0]=arrtime[0];
 for(i=1;i<n;i++)
 {
  sum=0;
  for(j=0;j<i;j++)
  sum=sum+bursttime[j];
  wt[i]=sum-arrtime[i];
 }

 for(i=0;i<n;i++)
 tat[i]=0;
 for(i=0;i<n;i++)
 {
  sum=0;
  for(j=0;j<=i;j++)
  sum=sum+bursttime[j];
  tat[i]=sum-arrtime[i];
 }

 for(i=0;i<n;i++)
 awt=awt+wt[i];

 for(i=0;i<n;i++)
 atat=atat+tat[i];

 awt=awt/n;
 atat=atat/n;
 printf("\narrtime   bursttime   waitingt  turnaroundt\n");
 for(i=0;i<n;i++)
 printf("\n%d           %d           %d           %d",arrtime[i],bursttime[i],wt[i],tat[i]);
 printf("\nAverage waiting time is %.2f",awt);
 printf("\nAverage turn around time is %.2f",atat);
 return 0;
}
