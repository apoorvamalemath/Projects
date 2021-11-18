#include<stdio.h>
#include<stdlib.h>
int main()
{
 int i,j,n,a[50],frame[10],no,k,avail,count=0;
system("clear");
printf("enter the no of pages\n");
scanf("%d",&n);
printf("enter the page no\n");
for(i=1;i<=n;i++)
scanf("%d",&a[i]);
printf("enter the no of frames\n");
scanf("%d",&no);
for(i=0;i<no;i++)
frame[i]=-1;
j=0;
printf("ref string\t page frames\n");
for(i=1;i<=n;i++)
{
printf("%d\t\t",a[i]);
avail=0;
for(k=0;k<no;k++)
if(frame[k]==a[i])
avail=-1;
if(avail==0)
{
frame[j]=a[i];
j=(j+1)%no;
count++;
for(k=0;k<no;k++)
printf("%d\t",frame[k]);
}
printf("\n");
}
printf("page fault is %d\n",count);
return 0;
}


